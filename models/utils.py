import numpy as np
import torch
from torch import nn


class CloneMixin:
    def clone(self):
        # noinspection PyArgumentList
        clone = self.__class__(*self.__args__)
        clone.load_state_dict(self.state_dict())
        return clone


class ParamUpdater:
    def __init__(self, src, destination, update_factor=1.0):
        self.src_model = src
        self.destination_model = destination
        self.update_factor = update_factor

    def __call__(self):
        for src_param, destination_parameter in zip(self.src_model.parameters(), self.destination_model.parameters()):
            updated_param = src_param.data * self.update_factor + destination_parameter * (1 - self.update_factor)
            destination_parameter.data.copy_(updated_param)


def flatten(x):
    return x.view(x.size()[0], -1)


def init_weights_xavier(model):
    for p_name, p_tensor in model.named_parameters():
        if '.weight' in p_name:
            nn.init.xavier_uniform_(p_tensor)


class ReusableTensor:
    def __init__(self, dtype, device):
        self._tensor = None
        self._dtype = dtype
        self._device = device

    def __call__(self, size):
        if self._tensor is None or self._tensor.size() != size:
            self._tensor = torch.zeros(
                *size, dtype=self._dtype, device=self._device)

        return self._tensor


class SARBuffer:
    def __init__(self):
        self._buffer = []

    def __len__(self):
        return len(self._buffer)

    def append(self, state, action, response):
        self._buffer.append((state, action, response))

    def get(self, n=None):
        """
        Takes first n entries from the current buffer (all if n is None)

        :param n: number of entries to be taken
        :return: entries (states, actions, responses)
        """
        n = len(self._buffer) if n is None else n
        states, actions, responses = zip(*self._buffer[:n])
        del self._buffer[:n]

        states = torch.from_numpy(np.stack(states)).cuda()
        actions = torch.as_tensor(actions).cuda()
        responses = torch.as_tensor(responses).cuda()

        return states, actions, responses


def make_lin_schedule(x_end, y_end):
    x = 0

    def lin_schedule():
        nonlocal x
        x += 1
        if x > x_end:
            return y_end
        return 1.0 + (y_end - 1.0) / x_end * x

    return lin_schedule
