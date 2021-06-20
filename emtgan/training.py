import click


input_features = [
    'x', 'y', 'z',
    'q',
    'ax', 'ay', 'az'
]

# set hyperparameters
batch_size = 1
discriminator_lr = 0.0005
generator_lr = 0.0005
num_epochs = 200
ensembles = 20
weight_decay = 0

betas = (
    0.5,
    0.999
)

lambda_adv = 1
lambda_cycle = 10
lambda_ident = 5
lambda_Q = 0
lambda_comp = 1e-3

# CC0: disable cycle-consistency constraint (like normal GAN training)
CC0 = False
variant = ''

if CC0:
    lambda_cycle = 0
    variant = 'CC0'
    
enable_scheduling = True


def model_error(G, x, y):
    input_branch_1, input_branch_2 = np.split(x, 2, 1)
    
    input_1 = np2torch(input_branch_1)
    input_2 = np2torch(input_branch_2)
    op_branch_1 = G(input_1)
    op_branch_2 = G(input_2)
    op_branch_1 = torch2np(torch.cat([input_1[:,:2], op_branch_1], 1))
    op_branch_2 = torch2np(torch.cat([input_2[:,:2], op_branch_2], 1))
    
    y_1, y_2 = np.split(y, 2, 1)
    dcap = np.linalg.norm(y_1 - y_2, axis=1)
    
    d = np.linalg.norm((unnormalize(op_branch_1) - unnormalize(op_branch_2))[:,:3], axis=1)
    return d - dcap


def model_MSE(G, x, y):
    d_err = model_error(G, x, y)
    err = d_err
    return np.sum(np.square(err)) / x.shape[0]


def train_iteration(epoch, iteration, D_cl, opt_D_cl, D_lc, opt_D_lc, G_cl, G_lc, opt_G, Xlab, Xcarm, ycarm):
    real, fake = make_labels_hard(Xlab.size(0))
    
    lab_1, lab_2 = torch.split(Xlab, len(input_features), 1)
    carm_1, carm_2 = torch.split(Xcarm, len(input_features), 1)
    
    ### train generators ###
    opt_G.zero_grad()
    
    fake_lab_1 = torch.cat([carm_1[:,:2], G_cl(carm_1)], 1)
    fake_lab_2 = torch.cat([carm_2[:,:2], G_cl(carm_2)], 1)
    
    fake_carm_1 = torch.cat([lab_1[:,:2], G_lc(lab_1)], 1)
    fake_carm_2 = torch.cat([lab_2[:,:2], G_lc(lab_2)], 1)
    
    ## adversarial loss ##
    # how well can G fool D?
    loss_D_cl_adv = bceloss(D_cl(torch.cat([fake_lab_1, fake_lab_2], 1)), real)
    loss_D_lc_adv = bceloss(D_lc(torch.cat([fake_carm_1, fake_carm_2], 1)), real)
    loss_adv = (loss_D_cl_adv + loss_D_lc_adv) / 2
    
    ## cycle loss ##
    # enforce cycle consistency
    recov_lab = torch.cat([fake_carm_1[:,:2], G_cl(fake_carm_1)], 1)
    recov_carm = torch.cat([fake_lab_1[:,:2], G_lc(fake_lab_1)], 1)
    
    loss_recov_lab = mse(recov_lab, lab_1)
    loss_recov_carm = mse(recov_carm, carm_1)
    loss_cycle = (loss_recov_lab + loss_recov_carm) / 2
    
    ## identity loss ##
    loss_ident_lab =  mse(lab_1, torch.cat([lab_1[:,:2], G_cl(lab_1)], 1))
    loss_ident_carm = mse(carm_1, torch.cat([carm_1[:,:2], G_lc(carm_1)], 1))
    loss_ident = (loss_ident_lab + loss_ident_carm) / 2
    
    d_fake = torch.norm(tensor_unnormalize(fake_lab_1)[:,:3] - tensor_unnormalize(fake_lab_2)[:,:3], 2, 1)
    y_1, y_2 = torch.split(ycarm, 3, 1)
    d_real = torch.norm(y_1 - y_2, 2, 1)
    loss_comp = mse(d_fake, d_real)
    
    ## total loss for both generators ##
    loss_G = lambda_adv * loss_adv + lambda_cycle * loss_cycle + lambda_ident * loss_ident + lambda_comp * loss_comp
    
    #if epoch >= 30:
    #torch.nn.utils.clip_grad_norm_(G_lc.parameters(), 1.0)
    #torch.nn.utils.clip_grad_norm_(G_cl.parameters(), 1.0)
    loss_G.backward()
    opt_G.step()
    
    
    real, fake = make_labels_soft(Xlab.size(0))
    
    ### train discriminators
    ## D_cl
    opt_D_cl.zero_grad()
    
    fake_lab_1 = torch.cat([carm_1[:,:2], G_cl(carm_1)], 1)
    fake_lab_2 = torch.cat([carm_2[:,:2], G_cl(carm_2)], 1)
    
    loss_real = bceloss(D_cl(Xlab), real) + bceloss(D_cl(Xcarm), fake)
    loss_fake = bceloss(D_cl(torch.cat([fake_lab_1, fake_lab_2], 1)), fake)
    
    loss_D_cl = (loss_real + loss_fake) / 3
    
    torch.nn.utils.clip_grad_norm_(D_cl.parameters(), 1.0)
    loss_D_cl.backward()
    opt_D_cl.step()
    
    ## D_lc
    opt_D_lc.zero_grad()
    
    fake_carm_1 = torch.cat([lab_1[:,:2], G_lc(lab_1)], 1)
    fake_carm_2 = torch.cat([lab_2[:,:2], G_lc(lab_2)], 1)
        
    loss_real = bceloss(D_lc(Xcarm), real) + bceloss(D_lc(Xlab), fake)
    loss_fake = bceloss(D_lc(torch.cat([fake_carm_1, fake_carm_2], 1)), fake)
    loss_D_lc = (loss_real + loss_fake) / 3
    
    torch.nn.utils.clip_grad_norm_(D_lc.parameters(), 1.0)
    loss_D_lc.backward()
    opt_D_lc.step()
    
    return dict(
        discriminator_CL=loss_D_cl,
        discriminator_LC=loss_D_lc,
        cycle=lambda_cycle * loss_cycle,
        adversarial=lambda_adv * loss_adv,
        ident=lambda_ident * loss_ident,
        comp=lambda_comp * loss_comp,
        generator=loss_G
    )


def train_emtcyclegan():
    val_losses = np.array([])
    min_val_loss_total = np.inf

    num_iterations = min(len(lab_dataloader), len(carm_dataloader))


    for model_num in range(ensembles):
        #### Discriminators ####
        ## D for c-arm --> lab conversion
        D_cl = CycleGANDiscriminatorNetwork().to(cuda)
        initialize_weights_normal(D_cl)
        opt_D_cl = optim.Adam(D_cl.parameters(), lr=discriminator_lr, betas=betas)
        
        ## D for lab --> c-arm conversion
        D_lc = CycleGANDiscriminatorNetwork().to(cuda)
        initialize_weights_normal(D_lc)
        opt_D_lc = optim.Adam(D_lc.parameters(), lr=discriminator_lr, betas=betas)
        
        #### Generators ####
        ## G for c-arm --> lab conversion
        G_cl = CycleGANGeneratorNetwork().to(cuda)
        initialize_weights_normal(G_cl)
        
        ## G for lab --> c-arm conversion
        G_lc = CycleGANGeneratorNetwork().to(cuda)
        initialize_weights_normal(G_lc)
        
        opt_G = optim.Adam(chain(G_lc.parameters(), G_cl.parameters()), lr=generator_lr, betas=betas)
        
        min_val_loss = np.inf
        min_val_index = 0
        
        hist_epoch = np.array([])
        hist_train_losses = {}
        hist_val_loss = np.array([])
        
        if enable_scheduling:
            sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=DecayLambda(num_epochs, 0, num_epochs // 2).step)
            sched_D_cl = optim.lr_scheduler.LambdaLR(opt_D_cl, lr_lambda=DecayLambda(num_epochs, 0, num_epochs // 2).step)
            sched_D_lc = optim.lr_scheduler.LambdaLR(opt_D_lc, lr_lambda=DecayLambda(num_epochs, 0, num_epochs // 2).step)
        
        ## adversarial training
        for epoch in range(num_epochs):
            train_losses = {}
            for iteration in range(num_iterations):
                lab_batch = next(iter(lab_dataloader))
                carm_batch = next(iter(carm_dataloader))

                Xlab = lab_batch['x'].float().to(cuda)
                Xcarm = carm_batch['x'].float().to(cuda)
                ycarm = carm_batch['y'].float().to(cuda)
                
                losses = train_iteration(
                    epoch,
                    iteration,
                    D_cl, opt_D_cl,
                    D_lc, opt_D_lc,
                    G_cl, G_lc, opt_G,
                    Xlab, Xcarm,
                    ycarm
                )
            for key, value in losses.items():
                if key not in train_losses:
                    train_losses[key] = np.array([])
                train_losses[key] = np.append(train_losses[key], np.mean(torch2np(losses[key])))
            update_loss_dict(hist_train_losses, train_losses)
            
            if enable_scheduling:
                sched_G.step()
                sched_D_cl.step()
                sched_D_lc.step()
            
            # average training loss
            hist_epoch = np.append(hist_epoch, epoch)
            
            # compute validation loss
            val_loss = model_MSE(G_cl, xval_N, yval)#np.mean(train_losses['generator'])
            hist_val_loss = np.append(hist_val_loss, val_loss)
            plot_history(
                hist_epoch,
                min_val_index=min_val_index,
                val_loss=hist_val_loss,
                max_epochs=num_epochs,
                **hist_train_losses
            )
            click.secho(f'ensemble: {model_num+1}/{ensembles}')
            click.secho(f'validation loss: {val_loss:.4f}    best: {min_val_loss:.4f}')
            if val_loss < min_val_loss:
                min_val_index = epoch
                torch.save(
                    { 'state_dict': G_cl.state_dict() },
                    os.path.join(model_dir, f'LabGAN{variant}_{model_num:03d}.pth')
                )
                torch.save(
                    { 'state_dict': D_cl.state_dict() },
                    os.path.join(model_dir, f'LabGAN{variant}_D_{model_num:03d}.pth')
                )
                min_val_loss = val_loss

