"""
OpenAI baselines replay buffer implementation. Modified to return PyTorch tensors.
"""

import random
from collections import namedtuple

import torch

ReplayTuple = namedtuple('ReplayTuple',
                         ['state_t', 'action_t', 'state_tp1',
                          'reward_t', 'terminal_t'])


def tuple_to_device(replay_tuple):
    return ReplayTuple(*[t.to('cuda:0', non_blocking=True) for t in replay_tuple])


class ReplayBuffer:
    def __init__(self, size, batch_size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.batch_size = batch_size

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        observations_t, actions, rewards, observations_tp1, terminal_flags = [], [], [], [], []
        for i in indices:
            obs_t, action, reward, obs_tp1, done = self._storage[i]
            observations_t.append(obs_t.torch())
            observations_tp1.append(obs_tp1.torch())

            actions.append(action)
            rewards.append(reward)
            terminal_flags.append(done)

        s_t = torch.stack(observations_t)
        s_tp1 = torch.stack(observations_tp1)

        a_t = torch.tensor(actions)
        r_t = torch.tensor(rewards)
        t_t = torch.tensor(terminal_flags, dtype=torch.uint8)
        return tuple_to_device(ReplayTuple(s_t, a_t, s_tp1, r_t, t_t))

    def sample(self):
        indices = [random.randint(0, len(self._storage) - 1) for _ in range(self.batch_size)]
        return self._encode_sample(indices)
