import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

### PyTorch shortcuts
mse = nn.MSELoss()
l1loss = nn.L1Loss()
bceloss = nn.BCELoss()
kld = nn.KLDivLoss()


def initialize_weights_normal(network):
    def init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    network.apply(init)
        
def initialize_weight_xavier(network):
    def init(m):
        nn.init.xavier_normal_(m.weight)
    network.apply(init)


def torch2np(data):
    return data.cpu().detach().numpy()

def np2torch(data):
    return torch.from_numpy(data).float().to(cuda)


if torch.cuda.is_available():
    cuda = torch.device('cuda')
    Tensor = torch.cuda.FloatTensor
else:
    cuda = torch.device('cpu')
    Tensor = torch.FloatTensor


LABEL_REAL = 0
LABEL_FAKE = 1

def make_labels_hard(N):
    real = Variable(Tensor(N, 1).fill_(LABEL_REAL), requires_grad=False)
    fake = Variable(Tensor(N, 1).fill_(LABEL_FAKE), requires_grad=False)
    return real, fake

def make_labels_soft(N, margin=0.1):
    real = Variable(Tensor(np.random.uniform(LABEL_REAL, margin, (N, 1))), requires_grad=False)
    fake = Variable(Tensor(np.random.uniform(LABEL_FAKE - margin, LABEL_FAKE, (N, 1))), requires_grad=False)
    return real, fake
