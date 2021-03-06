import json
import os


with open('config.json', 'r') as f:
    config = json.load(f)

root_dir = config.get('ROOT_DIR', '')
data_dir = os.path.join(root_dir, 'data')
model_dir = os.path.join(root_dir, 'saved_models')
os.makedirs(model_dir, exist_ok=True)

from itertools import chain
import numpy as np
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')

# torch
import torch
from torch import nn, optim
from torch.autograd import Variable

from scipy.optimize import minimize

import pydicom


def permute_unique(N: int):
    """
    Return permutations for point tuple indices; each permutation
    is unique.
    Example (N = 4):
    0 1
    0 2
    0 3
    1 2
    1 3
    2 3
    """
    assert N > 0
    pairs = None
    for i in range(N):
        for j in range(i + 1, N):
            if pairs is None:
                pairs = np.array([i, j])
            else:
                pairs = np.vstack((pairs, np.array([i, j])))
    return pairs


def _parse_model_section(lines: list):
    model_offsets = {}
    for line in lines:
        token = ''
        at_model = False
        at_offset = False
        model_coords = []
        offset_coords = []
        for c in line:
            if c == '(':
                if not at_model and not at_offset:
                    at_model = True
                elif not at_model and at_offset:
                    pass
            elif c == ')':
                pass
            elif c == ':':
                at_model = False
                at_offset = True
            elif c.isspace():
                pass
            elif c == ',':
                pass
            elif c.isdigit() or c in ('e', '.', '-'):
                token += c
                continue
            else:
                raise RuntimeError('unexpected character ' + c)
            if token:
                if at_model:
                    model_coords.append(float(token))
                elif at_offset:
                    offset_coords.append(float(token))
                token = ''
        if len(model_coords) != len(offset_coords):
            raise RuntimeError('invalid mapping of coords in MODEL section')
        modelpoint = tuple(model_coords)
        offset = offset_coords
        if modelpoint and offset:
            model_offsets[modelpoint] = np.array(offset)
    return model_offsets

def load_calibration(path: str):
    """
    Load calibration data from file.

    :param path: path to calibration file
    """
    sections = {}
    current_section = ''
    section_lines = []
    with open(path, 'r') as f:
        text = f.read()
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            idx = line.find('#')
            if idx != -1:
                line = line[:idx]
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                if current_section:
                    sections[current_section].extend(section_lines)
                    section_lines = []
                current_section = line[1:-1].upper()
                if current_section not in sections:
                    sections[current_section] = []
            elif not current_section:
                raise RuntimeError('line does not belong to any section')
            else:
                section_lines.append(line)
        if section_lines:
            sections[current_section].extend(section_lines)
    return _parse_model_section(sections['MODEL'])


calibration_map = load_calibration(os.path.join(root_dir, 'calibration.txt'))

class Sensor:
    def __init__(self):
        """
        """
        self.data = None

class Measurement:
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.sensors = []
        self.model = None

    def points(self, features):
        P = self.sensors[1].data[features].values
        return P

    def modelpoints(self, offset=False):
        P = self.model
        if offset:
            ref = self.sensors[0].data[['x', 'y', 'z']].median().values
            return P + ref
        return P

    def displacements(self, features, permute_func=permute_unique):
        """
        Return list of displacement vectors generated by a custom permutation.
        :param features: input feature vector
        """
        assert len(features) > 0
        assert len(self.sensors) >= 2
        n = len(features)
        points = self.sensors[1].data[features].values
        assert len(points) > 0
        permutations = permute_func(len(points))
        D = np.zeros((len(permutations), n * 2))
        for i, p in enumerate(permutations):
            D[i, 0:n] = points[p[0], 0:n]
            D[i, n:n*2] = points[p[1], 0:n]
        return D

    def model_displacements(self, permute_func=permute_unique):
        permutations = permute_func(len(self.model))
        M = np.zeros((len(permutations), 6))
        for i, p in enumerate(permutations):
            M[i, 0:3] = self.model[p[0], 0:3]
            M[i, 3:6] = self.model[p[1], 0:3]
        return M

    def __len__(self):
        return len(self.model)


def filter_df(df, nsamples):
    return df.groupby(df.index // nsamples).median()


class Dataset:
    def __init__(self, identifier):
        self.identifier = identifier
        self.measurements = []
        self.environments = set()

    def subset(self, environments=None, description=None):
        """
        :param description: regex that is matched to measurement descriptions
        """
        if environments is None and description is None:
            raise RuntimeError('invalid subset query')
        sub = Dataset(self.identifier + '_sub')
        inverse = Dataset(self.identifier + '_inv')
        regex = re.compile(description)
        for measurement in self.measurements:
            if environments and measurement.environment in environments:
                sub.add_measurement(measurement)
                continue
            if description and re.match(regex, measurement.description):
                sub.add_measurement(measurement)
                continue
            inverse.add_measurement(measurement)
        return sub, inverse

    def merge(self, dataset):
        merged = Dataset(self.identifier + '_' + dataset.identifier)
        merged.measurements = self.measurements
        merged.environments = self.environments
        merged.measurements.extend(dataset.measurements)
        merged.environments.update(dataset.environments)
        return merged

    def add_measurement(self, m: Measurement):
        self.measurements.append(m)
        self.environments.add(m.environment)

    def summary(self):
        s = '[DATASET] {}\n'.format(self.identifier)
        s += '{} measurements\n'.format(len(self.measurements))
        s += 'environments: {}\n'.format(','.join(self.environments))
        s += 'rmse: {}\n'.format(self.rmse())
        return s

    def displacements(self,
            features,
            permute_func=permute_unique,
            remove_duplicates=False,
            shuffle=False):
        """
        Return dataset displacements and corresponding model displacements.
        :param features: input feature vector
        :param permute_func: permutation function to generate displacements
        :param remove_duplicates: remove redundant measured displacements
        """
        assert len(features) > 0
        D = None
        M = None
        for measurement in self.measurements:
            d = measurement.displacements(features, permute_func)
            m = measurement.model_displacements(permute_func)
            if D is None:
                D = d
                M = m
            else:
                D = np.vstack((D, d))
                M = np.vstack((M, m))
        if remove_duplicates:
            _, idx = np.unique(D, axis=0, return_index=True)
            if shuffle:
                np.random.shuffle(idx)
            D = D[idx, :]
            M = M[idx, :]
        return D, M

    def points(self, features):
        P = None
        for measurement in self.measurements:
            p = measurement.points(features)
            if P is None:
                P = p
            else:
                P = np.vstack((P, p))
        return P

    def modelpoints(self, offset=False):
        P = None
        for measurement in self.measurements:
            p = measurement.modelpoints(offset)
            if P is None:
                P = p
            else:
                P = np.vstack((P, p))
        return P

    def errors(self):
        D, M = self.displacements(['x', 'y', 'z'])
        d_D = np.linalg.norm(D[:, 0:3] - D[:, 3:6], axis=1)
        d_M = np.linalg.norm(M[:, 0:3] - M[:, 3:6], axis=1)
        E = d_D - d_M
        return E

    def mse(self):
        E = self.errors()
        mse = np.sum(np.square(E)) / len(E)
        return mse

    def rmse(self):
        return np.sqrt(self.mse())


def load_measurement(directory: str):
    assert len(directory) > 0
    m = Measurement(directory)
    m_meta = json.load(open(os.path.join(directory, 'measurement.json'), 'r'))
    m.__dict__.update(m_meta)
    points = pd.read_csv(os.path.join(directory, 'points.csv'), header=None)
    m.model = points.values * np.array([8.0, 8.0, 9.6])

    def get_calibrated_point(x):
        return calibration_map.get((x[0], x[1], x[2]), (0, 0, 0))

    offsets = np.apply_along_axis(get_calibrated_point, axis=1, arr=m.model)
    m.model -= offsets
    nsamples = m.samples_per_point

    # TODO @henry extend this to up to 4 sensors
    s0 = Sensor()
    with open(os.path.join(directory, 'sensor_0.json'), 'r') as f:
        s0_meta = json.load(f)
        s0.__dict__.update(s0_meta)
    data = pd.read_csv(os.path.join(directory, 'sensor_0.csv'))
    s0.data = filter_df(data, nsamples)

    s1 = Sensor()
    with open(os.path.join(directory, 'sensor_1.json'), 'r') as f:
        s1_meta = json.load(f)
        s1.__dict__.update(s1_meta)
    data = pd.read_csv(os.path.join(directory, 'sensor_1.csv'))
    sx = data['x'].std()
    sy = data['y'].std()
    sz = data['z'].std()
    sq = data['q'].std()
    s1.data = filter_df(data, nsamples)
    s1.data['rq'] = s0.data['q']
    s1.data['sx'] = data['x'].groupby(data.index // nsamples).std()
    s1.data['sy'] = data['y'].groupby(data.index // nsamples).std()
    s1.data['sz'] = data['z'].groupby(data.index // nsamples).std()
    s1.data['sq'] = data['q'].groupby(data.index // nsamples).std()
    m.sensors.append(s0)
    m.sensors.append(s1)
    return m 


def load_dataset(directory: str):
    assert len(directory) > 0
    dataset = Dataset(os.path.basename(directory))
    for measurement_dir in os.listdir(directory):
        if not measurement_dir.startswith('measurement'):
            continue
        m = load_measurement(os.path.join(directory, measurement_dir))
        dataset.add_measurement(m)
    return dataset


def flatten_first_dim(x, lmax=np.inf):
    temp = []
    for i in range(len(x)):
        length = min(len(x[i]), lmax)
        for j in range(length):
            temp.append(x[i][j])
    return np.asarray(temp)


def load_datasets(list_n, input_features=['x', 'y', 'z']):
    X = []
    Y = []
    for carm in list_n:
        print(f'{carm}.....', end='')
        carm_path = os.path.join(data_dir, carm)
        dataset = load_dataset(carm_path)
        X.append(dataset.displacements(input_features)[0])
        Y.append(dataset.displacements(input_features)[1])
        print('done')
    return X, Y


def dataset_bounds(dataset, labels, input_features):
    dim = len(input_features)
    Min = np.min((dataset.min(axis=0)[:dim], dataset.min(axis=0)[dim:]), axis=0)
    Max = np.max((dataset.max(axis=0)[:dim], dataset.max(axis=0)[dim:]), axis=0)
    labelMin = np.min(labels)
    labelMax = np.max(labels)
    return Min, Max, labelMin, labelMax


class DecayLambda():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), 'Decay must start before the training session ends!'
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
