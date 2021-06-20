LABEL_REAL = 0
LABEL_FAKE = 1

def make_labels_hard(N: int):
    real = Variable(Tensor(N, 1).fill_(LABEL_REAL), requires_grad=False)
    fake = Variable(Tensor(N, 1).fill_(LABEL_FAKE), requires_grad=False)
    return real, fake

def make_labels_soft(N: int, margin=0.1):
    real = Variable(Tensor(np.random.uniform(LABEL_REAL, margin, (N, 1))), requires_grad=False)
    fake = Variable(Tensor(np.random.uniform(LABEL_FAKE - margin, LABEL_FAKE, (N, 1))), requires_grad=False)
    return real, fake
