"""
Low level building-blocks for the various architectures.
"""

from collections import OrderedDict

from torch import nn
from models.utils import flatten


class FlattenModule(nn.Module):
    def forward(self, input):
        return flatten(input)


class ConvRelu(nn.Sequential):
    def __init__(self, conv_sizes):
        layers = OrderedDict()

        for i, conv_size in enumerate(conv_sizes):
            layers[f'conv{i}'] = nn.Conv2d(*conv_size)
            layers[f'relu{i}'] = nn.ReLU()

        layers['flatten'] = FlattenModule()
        super().__init__(layers)


class LinearRelu(nn.Sequential):
    def __init__(self, layer_sizes):
        layers = OrderedDict()

        input_size = layer_sizes[0]
        for i, layer_size in enumerate(layer_sizes[1:]):
            layers[f'lin{i}'] = nn.Linear(input_size, layer_size)
            layers[f'relu{i}'] = nn.ReLU()
            input_size = layer_size

        super().__init__(layers)
