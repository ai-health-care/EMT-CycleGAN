import os

import numpy as np
import torch
import torch.utils.data as data

from .common import flatten_first_dim, load_dataset, load_datasets, data_dir, dataset_bounds
from .utils import *

batch_size = 1

input_features = [
    'x', 'y', 'z',
    'q',
    'ax', 'ay', 'az',
    'rq'
]


list_datasets_lab = [
    '3d/lab'
]

list_datasets_carm = [
    '3d/c-arm-9',
    '3d/c-arm-13',
    '3d/c-arm-16',
]

list_datasets_val = [
    '3d/c-arm-12',
    '3d/c-arm-15'
]

list_datasets_test = [
    '3d/c-arm-10',
    '3d/c-arm-11',
    '3d/c-arm-14'
]


test_datasets = {}

x, y = load_datasets(list_datasets_lab, input_features)
xlab = flatten_first_dim(x)
ylab = flatten_first_dim(y)

x, y = load_datasets(list_datasets_carm, input_features)
xcarm = flatten_first_dim(x)
ycarm = flatten_first_dim(y)

x, y = load_datasets(list_datasets_val, input_features)
xval = flatten_first_dim(x)
yval = flatten_first_dim(y)

x = []
y = []
for carm in list_datasets_test:
    print(f'{carm}.....', end='')
    carm_path = os.path.join(data_dir, carm)
    dataset = load_dataset(carm_path)
    points, gt = dataset.displacements(input_features)
    test_datasets[carm] = (points, gt)
    x.append(points)
    y.append(gt)
    print('done')
xtest = flatten_first_dim(x)
ytest = flatten_first_dim(y)


lab_min, lab_max, lab_ymin, lab_ymax = dataset_bounds(xlab, ylab, input_features)
carm_min, carm_max, carm_ymin, carm_ymax = dataset_bounds(xcarm, ycarm, input_features)
val_min, val_max, val_ymin, val_ymax = dataset_bounds(xval, yval, input_features)
test_min, test_max, test_ymin, test_ymax = dataset_bounds(xtest, ytest, input_features)

min_vec = np.min((lab_min, carm_min, test_min, val_min), axis=0)
max_vec = np.max((lab_max, carm_max, test_max, val_max), axis=0)
min_vec2 = np.append(min_vec, min_vec)
max_vec2 = np.append(max_vec, max_vec)

min_y = np.min((lab_ymin, carm_ymin, test_ymin, val_ymin), axis=0)
max_y = np.max((lab_ymax, carm_ymax, test_ymax, val_ymax), axis=0)

xlab_N = (xlab - min_vec2) / (max_vec2 - min_vec2)
xcarm_N = (xcarm - min_vec2) / (max_vec2 - min_vec2)
xtest_N = (xtest - min_vec2) / (max_vec2 - min_vec2)
xval_N = (xval - min_vec2) / (max_vec2 - min_vec2)

class PointDataset(data.Dataset):
    def __init__(self, x, gt):
        self.x = x
        self.gt = gt
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        x = self.x[index]
        gt = self.gt[index]
        
        return {
            'x': x.astype('float64'),
            'gt': gt.astype('float64')
        }

lab = PointDataset(xlab_N, ylab)
lab_2layers = PointDataset(xlab_N[xlab[:,2] > -10], ylab[xlab[:,2] > -10])
carm = PointDataset(xcarm_N, ycarm)

lab_dataloader = torch.utils.data.DataLoader(lab, batch_size=batch_size, shuffle=True)
lab_2layers_dataloader = torch.utils.data.DataLoader(lab_2layers, batch_size=batch_size, shuffle=True)
carm_dataloader = torch.utils.data.DataLoader(carm, batch_size=batch_size, shuffle=True)


T_max_vec = torch.from_numpy(max_vec).float().to(cuda)
T_min_vec = torch.from_numpy(min_vec).float().to(cuda)

def normalize(p):
    return (p - min_vec[:p.shape[1]]) / (max_vec - min_vec)[:p.shape[1]]

def unnormalize(p):
    return p * (max_vec - min_vec)[:p.shape[1]] + min_vec[:p.shape[1]]

def tensor_normalize(T, dim=0):
    M = torch.sub(T, T_min_vec[:T.size(1)])
    return torch.div(M, T_max_vec[:T.size(1)] - T_min_vec[:T.size(1)])

def tensor_unnormalize(T):
    M = torch.mul(T, T_max_vec[:T.size(1)] - T_min_vec[:T.size(1)])
    return torch.add(M, T_min_vec[:T.size(1)])
