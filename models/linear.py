"""
Implements Bayesian linear model for use in Successor Uncertainties.
"""

import torch
from torch import nn


class OnlineBayesLinReg(nn.Module):
    _dtype = torch.float64
    _device = 'cuda'

    def __init__(self, input_size, out_std, bias=False, use_mean=False, eps=1e-6, decay_factor=0.0):
        super().__init__()

        self.dim = input_size + int(bias)
        self._bias, self._eps = bias, eps
        self._decay_factor = decay_factor
        self.out_std = out_std

        self._use_mean = use_mean

        self._identity_mat = torch.eye(self.dim, device=self._device, dtype=self._dtype)
        self._scaled_mean = nn.Parameter(torch.zeros(
            self.dim, device=self._device, dtype=self._dtype), requires_grad=False)
        self._scaled_precision = nn.Parameter(torch.zeros(
            self.dim, self.dim, device=self._device, dtype=self._dtype), requires_grad=False)

        self._scaled_precision += self._identity_mat

    @staticmethod
    def _prepend_unit_vector(observations):
        ones_vector = torch.ones(*observations.size()[:-1], 1, device=observations.device, dtype=observations.dtype)
        return torch.cat([ones_vector, observations], dim=-1)

    @staticmethod
    def preprocess_inputs(observations, obs_dim, normalise, bias, dtype=None):
        observations = observations.to(OnlineBayesLinReg._device, dtype=dtype or OnlineBayesLinReg._dtype)

        if observations.dim() == obs_dim:
            observations = observations.unsqueeze(0)
        if normalise:
            observations /= observations.norm(dim=-1, keepdim=True)
        if bias:
            observations = OnlineBayesLinReg._prepend_unit_vector(observations)

        return observations

    def _get_svd_scaled_covar(self):

        correction = (1.0 + self._scaled_precision.diag().abs().mean()) * self._eps * self._identity_mat
        u, s, v = (self._scaled_precision + correction).svd()

        # correct for numerical errors and convert to factorisation of the pseudo-inverse
        s[s != 0.0] = 1.0 / s[s != 0.0]
        s[s < self._eps] = self._eps

        return u, s, v

    def _get_scaled_covar(self):
        u, s, v = self._get_svd_scaled_covar()
        return u @ s.diag() @ u.t()  # if covar is psd, u == v (if not we correct by assuming u == v)

    def get_covar(self):
        covar = self.out_std ** 2 * self._get_scaled_covar()
        return covar + (1.0 + covar.diag().abs().mean()) * self._eps * self._identity_mat

    def get_mean(self):
        return self._get_scaled_covar() @ self._scaled_mean

    def forward(self, observations):
        observations = OnlineBayesLinReg.preprocess_inputs(observations, 1, False, self._bias)
        return observations @ self.get_mean()

    def _update_mean(self, observations, responses):
        if self._decay_factor:
            exponent = torch.arange(len(responses) - 1, -1, -1, device=self._device, dtype=self._dtype)
            decay_mean = (1.0 - self._decay_factor) ** exponent
            decayed_responses = responses * decay_mean

            self._scaled_mean *= (1.0 - self._decay_factor) ** len(responses)
        else:
            decayed_responses = responses
        self._scaled_mean += observations.t() @ decayed_responses

    def _update_precision(self, observations, responses):
        if self._decay_factor:
            exponent = (torch.arange(len(responses) - 1, -1, -1, device=self._device, dtype=self._dtype) / 2.0)
            precision_decay = (1.0 - self._decay_factor) ** exponent
            decayed_observations = observations * precision_decay.unsqueeze(-1)

            self._scaled_precision *= (1.0 - self._decay_factor) ** len(responses)
        else:
            decayed_observations = observations
        self._scaled_precision += decayed_observations.t() @ decayed_observations

    def update(self, observations, responses):
        if len(observations) == 0:
            return

        with torch.no_grad():
            observations = OnlineBayesLinReg.preprocess_inputs(observations, 1, False, self._bias)
            responses = responses.to(self._device, dtype=self._dtype)

            if self._use_mean:
                self._update_mean(observations, responses)
            self._update_precision(observations, responses)

    def sample_weights(self, n=1, centred=False):
        """
        Samples weights from current posterior

        :param n: number of samples
        :param centred: if True predictive mean is ignored

        :return: n samples, one per row
        """
        u, s, v = self._get_svd_scaled_covar()
        scaled_covar_factor = u @ s.sqrt().diag()
        scaled_covar = scaled_covar_factor @ scaled_covar_factor.t()

        mean = torch.zeros(self.dim, device=self._device, dtype=self._dtype) if centred \
            else scaled_covar @ self._scaled_mean

        noise = torch.randn(n, self.dim, device=self._device, dtype=self._dtype) @ (
                self.out_std * scaled_covar_factor).t()

        return mean + noise


class SuccessorUncertaintyModel(nn.Module):
    def __init__(self, input_size, out_std=1.0, bias=False, zero_mean_weights=False,
                 eps=1e-6,
                 decay_factor=0.0):
        super().__init__()
        n_models = 1
        self.obs_dims = 2

        self.input_size = input_size
        self.zero_mean_weights = zero_mean_weights
        self.out_std = out_std

        self._bias = bias
        self._linear_models = [OnlineBayesLinReg(input_size=self.input_size, out_std=self.out_std,
                                                 bias=self._bias, eps=eps, decay_factor=decay_factor) for _ in
                               range(n_models)]
        self._weight_sample = self._get_weight_sample()

    def _preprocess_inputs(self, observations):
        return OnlineBayesLinReg.preprocess_inputs(observations, self.obs_dims, False, self._bias,
                                                   dtype=observations.dtype)

    def _get_weight_sample(self, n=1):
        """
        Samples n x [no. of features] matrix of weights
        """
        return self._linear_models[0].sample_weights(n, centred=self.zero_mean_weights).float()

    def _predict(self, observations, weights):
        return (self._preprocess_inputs(observations) @ weights.t()).permute(2, 0, 1)

    def resample_weights(self):
        self._weight_sample = self._get_weight_sample()

    def sample_responses(self, observations, n=1, reuse_weights=False):
        """
        Samples n x [no. of observations] x [no. of actions] matrix of responses
        """
        weights = self._weight_sample if reuse_weights else self._get_weight_sample(n)
        mean = self._predict(observations, weights)

        return torch.normal(mean, self.out_std * torch.ones_like(mean))

    def forward(self, observations):
        raise self.sample_responses(observations)

    def update(self, indices, observations, responses):
        observations = observations.gather(
            1, indices.long().view(-1, 1, 1).expand(-1, -1, observations.size()[-1])).squeeze(1)
        self._linear_models[0].update(observations.double(), responses.double())

    def get_covar(self):
        return self._linear_models[0].get_covar().float()

    def get_marginal_covar(self, observations):
        observations = self._preprocess_inputs(observations).float()
        return observations @ self.get_covar() @ observations.transpose(-1, -2)
