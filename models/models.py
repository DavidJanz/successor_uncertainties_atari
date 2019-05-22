"""
Implements QNetwork and SFQNetwork objects, which wrap the networks and loss functions required to run Q-learning
and the Successor Uncertainties algorithm respectively.
"""
from abc import ABC

from torch import nn

import losses_functional
from .architectures import Q, SF
from .utils import ParamUpdater


class _TD(nn.Module, ABC):
    def __init__(self, action_size):
        super().__init__()
        self.action_size = action_size
        self.named_loss_dict = {}
        self._policy = None

    def loss(self, replay_tuple):
        raise NotImplementedError()

    def register_policy(self, policy):
        self._policy = policy

    def local_embedding(self, **kwargs):
        return self.global_embedding(**kwargs)

    def global_embedding(self, x):
        raise NotImplementedError()


class QNetwork(_TD):
    def __init__(self, action_size, hidden_size, bias_out=False):
        super().__init__(action_size)

        self.q_network = Q(self.action_size, hidden_size, bias_out=bias_out)
        self.q_network_target = self.q_network.clone()

        self.update_params = ParamUpdater(self.q_network, self.q_network_target)

    def loss(self, replay_tuple):
        ql = losses_functional.q_loss(replay_tuple, self.q_network.q_fn, self.q_network_target.q_fn, 0.99)
        self.named_loss_dict = {'q_loss': float(ql.item()), 'total': float(ql.item())}
        return ql

    def forward(self, input):
        self.q_network(input)

    def q_fn(self, **kwargs):
        return self.q_network.q_fn(**kwargs)

    def global_embedding(self, **kwargs):
        return self.q_network_target.global_embedding(**kwargs)


class SFQNetwork(_TD):
    def __init__(self, action_size, hidden_size, successor_size, bias_out=False):
        super().__init__(action_size)
        self.successor_size = successor_size

        self.sf = SF(action_size, hidden_size, self.successor_size, bias_out=bias_out)
        self.sf_target = self.sf.clone()

        self.update_params = ParamUpdater(self.sf, self.sf_target)

    def loss(self, replay_tuple):
        loss_sf, loss_q, loss_rwd = \
            losses_functional.sfq_loss(
                replay_tuple, self.sf.featuriser, self.sf.local_embedding, self.weight(), self.bias(),
                self.sf.global_embedding, self.sf_target.global_embedding, self._policy, 0.99)

        total = loss_rwd
        self.named_loss_dict = {'reward': float(loss_rwd.item())}

        if loss_sf is not None:
            total += loss_sf
            self.named_loss_dict['successor_features'] = float(loss_sf.item())

        if loss_q is not None:
            total += loss_q
            self.named_loss_dict['q_loss'] = float(loss_q.item())

        self.named_loss_dict['total'] = float(total.item())
        return total

    def forward(self, input):
        self.sf(input)

    def q_fn(self, **kwargs):
        return self.sf.q_fn(**kwargs)

    def global_embedding(self, **kwargs):
        return self.sf.global_embedding(**kwargs)

    def local_embedding(self, **kwargs):
        return self.sf.local_embedding(**kwargs)

    def weight(self):
        return self.sf.reward_net.weight()

    def bias(self):
        return self.sf.reward_net.bias()
