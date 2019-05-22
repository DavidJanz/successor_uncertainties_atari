"""
Implements various reinforcement learning policies. UncertaintyPolicy is utilised by Successor Uncertainties.
"""


import random

import numpy as np
import torch
from torch import nn

from models.utils import SARBuffer, make_lin_schedule
from .utils import ReusableTensor


class _Policy(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.action_size = action_size
        self.softness = None
        self.p_buffer = []
        self.q_val_buffer = []
        self.action_buffer = []

    def start_new_episode(self):
        pass

    def update(self, states, actions, rewards):
        pass

    def forward(self, state):
        state = state.unsqueeze(0) if state.dim() == 3 else state
        action = int(self.sample(state).item())
        self.action_buffer.append(action)
        return action

    def sample(self, state):
        action, q_val = self._sample(state)
        self.q_val_buffer.append(q_val)
        return action

    def _sample(self, state):
        raise NotImplementedError()

    def expectation(self, state_embedding):
        p_values = self._expectation(state_embedding)
        self.p_buffer.append(p_values.detach().cpu())
        return p_values

    def _expectation(self, state_embedding):
        raise NotImplementedError()

    def get_stats(self):
        p_stats = {}
        if self.p_buffer:
            p_values = torch.cat(self.p_buffer)
            top_p, second_p = p_values.topk(k=2)[0].mean(0).unbind()
            bottom_p = p_values.min(1)[0].mean()
            p_stats = {'1st_p': float(top_p.item()), '2nd_p': float(second_p.item()),
                       'last_p': float(bottom_p.item())}
            self.p_buffer.clear()

        avg_q_val = np.mean(self.q_val_buffer)
        self.q_val_buffer.clear()

        action_history = self.action_buffer.copy()
        self.action_buffer.clear()

        return p_stats, avg_q_val, action_history

    def get_covariance(self):
        return None


class UniformPolicy(_Policy):
    def __init__(self, action_size):
        super().__init__(action_size)
        self.softness = 1 / self.action_size

    def _sample(self, state):
        return torch.randint(0, self.action_size, (1,), dtype=torch.long).cuda(), 0.0

    def _expectation(self, state):
        return torch.ones(self._policy_size(state), dtype=torch.float32).cuda() / self.action_size


class GreedyPolicy(_Policy):
    def __init__(self, action_size, q_fn):
        super().__init__(action_size)
        self._q_fn = q_fn
        self.softness = 0.0

    def _sample(self, state):
        max_val, action = self._q_fn(state=state).max(-1)
        return action, float(max_val.item())

    def _expectation(self, state):
        p = torch.zeros_like(state)
        j = self._q_fn(state).argmax(-1)
        p[(range(len(j)), j)] = 1
        return p.squeeze(0) if len(p) == 1 else p


class EpsGreedyPolicy(_Policy):
    def __init__(self, action_size, q_fn, decay_end_step, eps_final):
        super().__init__(action_size)
        self._q_fn = q_fn
        self.eps_fn = make_lin_schedule(decay_end_step, eps_final)
        self.eps = 1.0
        self._last_q_val = 0.0

    def _sample(self, state):
        self.eps = self.eps_fn()
        self.softness = self.eps / (self.action_size - 1)
        if random.random() < self.eps:
            return torch.randint(0, self.action_size, (1,), dtype=torch.long), self._last_q_val
        else:
            max_val, action = self._q_fn(state=state).max(-1)
            self._last_q_val = float(max_val.item())
            return action, self._last_q_val

    def _expectation(self, state):
        p = torch.ones_like(state).cuda() * self.eps / self.action_size
        j = self._q_fn(state=state).argmax(-1)
        p[(range(len(j)), j)] += 1 - self.eps
        return p.squeeze(0) if len(p) == 1 else p


class UncertaintyPolicy(_Policy):
    def __init__(self, action_size, uncertainty_model, local_embedding, global_embedding,
                 q_fn=None, q_responses=False, resample_every=0, batch_size=500):
        super().__init__(action_size)

        self.local_embedding, self.global_embedding = local_embedding, global_embedding
        self.uncertainty_model, self.q_fn = uncertainty_model, q_fn
        self.q_responses = q_responses

        self.buffer = SARBuffer()

        self._sample_tensor = ReusableTensor(torch.float32, 'cuda')
        self._argmax_tensor = ReusableTensor(torch.float32, 'cuda')

        self._resample_every = resample_every
        self._steps_since_resample = 0
        self._batch_size = batch_size

    def _update_and_resample(self):
        if self.buffer:
            with torch.no_grad():
                while len(self.buffer) > 0:
                    states, actions, responses = self.buffer.get(self._batch_size)
                    state_embeddings = self.local_embedding(state=states)

                    if self.q_responses:
                        responses = self.q_fn(state_embedding=state_embeddings)
                        responses = responses.gather(-1, actions.long().view(-1, 1)).squeeze(-1)

                    self.uncertainty_model.update(actions, state_embeddings, responses)
                    del states, actions, responses, state_embeddings  # releases GPU memory

                self.uncertainty_model.resample_weights()
        self._steps_since_resample = 0

    def start_new_episode(self):
        if not self._resample_every:
            self._update_and_resample()

    def update(self, state, action, response):
        self.buffer.append(state, action, response)

    def _sample(self, states):
        self._steps_since_resample += 1
        if self._resample_every and self._steps_since_resample == self._resample_every:
            self._update_and_resample()

        with torch.no_grad():
            state_embedding = self.global_embedding(state=states)
            sample = self.uncertainty_model.sample_responses(state_embedding, 1, reuse_weights=True)
            if self.q_fn is not None:
                sample += self.q_fn(state_embedding=state_embedding)
            max_val, action = sample.max(-1)
            return action, float(max_val.item())

    def _expectation(self, state_embedding, n_mc_samples=100):
        marginal_covariance = self.uncertainty_model.get_marginal_covar(state_embedding)
        variances = torch.diagonal(marginal_covariance, dim1=-1, dim2=-2)

        samples = variances * self._sample_tensor((n_mc_samples, *variances.size())).normal_()
        if self.q_fn is not None:
            samples += self.q_fn(state_embedding=state_embedding).to(samples.device)

        argmax_samples = self._argmax_tensor(samples.size()).zero_().scatter_(
            -1, samples.argmax(-1).unsqueeze(-1), 1)
        return argmax_samples.mean(0)

    def get_covariance(self):
        return self.uncertainty_model.get_covariance()
