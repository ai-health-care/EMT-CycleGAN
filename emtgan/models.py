import torch
from torch import nn


class CycleGANGeneratorNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CycleGANGeneratorNetwork, self).__init__(*args, **kwargs)
        leak = kwargs.pop('G_leak', 0.01)
        self.input_size = kwargs.pop('G_input_size', 8)
        self.units = kwargs.pop('G_units', 16)
        self.dropout = kwargs.pop('G_dropout', 0)
        self.num_layers = kwargs.pop('G_layers', 1)
        activation = nn.LeakyReLU(leak)
        self.output_size = self.input_size - 2
        self.first_layer = nn.Sequential(nn.Linear(self.input_size, self.units), activation, nn.Dropout(self.dropout))
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.units, self.units), activation, nn.Dropout(self.dropout))
            for _ in range(self.num_layers)
        ])
        self.final_layer = nn.Sequential(nn.Linear(self.units, self.output_size))

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


class CycleGANDiscriminatorNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CycleGANDiscriminatorNetwork, self).__init__(*args, **kwargs)
        self.input_size = kwargs.pop('D_input_size', 8)
        self.num_layers = kwargs.pop('D_layers', 2)
        self.first_layer = nn.Sequential(nn.Linear(self.input_size * 2, 16), nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU(0.2))
            for _ in range(self.num_layers)
        ])
        self.final_layer = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        
    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.final_layer(x)
        return x 
