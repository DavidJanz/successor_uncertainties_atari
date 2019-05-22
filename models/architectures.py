"""
Network architectures for Q-learning and Successor Uncertainties.
"""


from torch import nn

from .modules import ConvRelu, LinearRelu
from .utils import CloneMixin, init_weights_xavier

"""
Architecture defaults as from Mnih DQN Nature paper. These same parameters are used in Successor Uncertainties.
"""
_default_conv_sizes = [(4, 32, 8, 4),
                       (32, 64, 4, 2),
                       (64, 64, 3, 1)]

_default_feature_size = 3136


class _R(nn.Module):
    """
    Implements a reward predictor. Maps flattened output of conv network to rewards.
    """

    def __init__(self, action_size, hidden_sizes, successor_size, bias_out, hidden_net):
        super().__init__()
        self.action_size, self.successor_size = action_size, successor_size

        self._hidden_net = hidden_net
        self._embed_net = LinearRelu([hidden_sizes, successor_size * action_size])

        self._linear = nn.Linear(successor_size, 1, bias=bias_out)

    def local_embedding(self, x):
        embedding = self._embed_net(self._hidden_net(x))
        norm = embedding.norm(dim=-1, keepdim=True) + 1e-6
        return (embedding / norm).view(-1, self.action_size, self.successor_size)

    def linear(self, x):
        return self._linear(x).squeeze(-1)

    def forward(self, x):
        return self.linear(self.local_embedding(x))

    def weight(self):
        return self._linear.weight

    def bias(self):
        return self._linear.bias


class _Psi(nn.Module):
    """
    Implements successor feature predictor. Maps flattened output of conv network to successor features.
    """
    def __init__(self, action_size, hidden_size, successor_size, hidden_net):
        super().__init__()
        self.__args__ = (action_size, hidden_size, successor_size)
        self._action_size = action_size
        self.successor_size = successor_size

        self._hidden_net = hidden_net
        self._embed_net = LinearRelu([hidden_size, successor_size * (action_size + 1)])

    def forward(self, x):
        embedding = self._embed_net(self._hidden_net(x))
        output = embedding.view(-1, self._action_size + 1, self.successor_size)
        return output[:, :self._action_size] + output[:, -1].view(-1, 1, self.successor_size)


class SF(nn.Module, CloneMixin):
    """
    Wrapper holding:
    1) featuriser, which maps states to flattened conv network output
    2) hidden network, mapping flattened conv output to a hidden layer
    3) reward network, which maps flattened conv output to state-action expected rewards using hidden network
    4) psi network, which maps flattened conv output to state-action expected successor features using hidden network
    Note that 3) and 4) share elements 1) and 2).
    """

    def __init__(self, action_size, hidden_size, successor_size, bias_out):
        super().__init__()
        self.__args__ = (action_size, hidden_size, successor_size, bias_out)
        self.featuriser = ConvRelu(_default_conv_sizes)

        hidden = LinearRelu([_default_feature_size, hidden_size])
        self.psi = _Psi(action_size, hidden_size, successor_size, hidden)
        self.reward_net = _R(action_size, hidden_size, successor_size, bias_out, hidden)

        init_weights_xavier(self)

    def forward(self, x):
        hidden = self.featuriser(x)
        embedding = self.post_featuriser(hidden)
        return self.linear(embedding)

    def local_embedding(self, *, state=None, state_features=None):
        if state is not None:
            state_features = self.featuriser(state)
        return self.reward_net.local_embedding(state_features)

    def global_embedding(self, *, state=None, state_features=None):
        if state is not None:
            state_features = self.featuriser(state)
        return self.psi(state_features)

    def q_fn(self, *, state=None, state_embedding=None):
        if state is not None:
            state_embedding = self.global_embedding(state=state)
        return self.reward_net.linear(state_embedding)

    def reward(self, x):
        return self.reward_net(self.featuriser(x))


class Q(nn.Module, CloneMixin):
    def __init__(self, action_size, hidden_size, bias_out):
        super().__init__()
        self.__args__ = (action_size, hidden_size, bias_out)

        self.featuriser = ConvRelu(_default_conv_sizes)
        self.post_featuriser = LinearRelu([_default_feature_size, hidden_size])
        self.linear = nn.Linear(hidden_size, action_size, bias=bias_out)

        init_weights_xavier(self)

    def forward(self, x):
        hidden = self.featuriser(x)
        embedding = self.post_featuriser(hidden)
        return self.linear(embedding)

    def q_fn(self, *, state=None, state_embedding=None):
        if state is not None:
            state_embedding = self.global_embedding(state=state)
        return self.linear(state_embedding)

    def global_embedding(self, *, state=None, state_features=None):
        if state is not None:
            state_features = self.featuriser(state)
        return self.post_featuriser(state_features)

    def get_global_embedding_dim(self):
        return self.post_featuriser.out_features
