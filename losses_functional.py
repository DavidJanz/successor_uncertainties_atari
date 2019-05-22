import torch
from torch.nn import functional


def _select(tensor, index, dim=-1):
    if tensor.dim() == 2:
        i = index.view(-1, 1) if index.dim() < 2 else index
    else:
        i = index.view(-1, 1, 1).expand(-1, -1, tensor.size()[-1])
    return tensor.gather(dim, i).squeeze(dim)


def _td_loss(x_t, x_diff_t, x_tp1, terminal, discount_factor):
    x_target = x_diff_t + (1 - terminal.float()) * discount_factor * x_tp1
    return functional.smooth_l1_loss(x_t, x_target)


def q_loss(replay_tuple, q_network, q_network_target, discount_factor):
    q_t_action = _select(q_network(state=replay_tuple.state_t), replay_tuple.action_t)

    with torch.no_grad():
        # double Q-learning, use live network to pick argmax, target network to assign value
        argmax_idx = q_network(state=replay_tuple.state_tp1).argmax(-1)
        q_tp1_max = _select(q_network_target(state=replay_tuple.state_tp1), argmax_idx)
    return _td_loss(q_t_action, replay_tuple.reward_t, q_tp1_max, replay_tuple.terminal_t, discount_factor)


def sfq_loss(replay_tuple, featuriser, local_embedding, weight_out, bias_out,
             global_embedding, global_embedding_target, policy, discount_factor):
    """
    Successor Features loss as defined in paper.

    :param replay_tuple: batch replay tuple for which loss is evalauted
    :param featuriser: conv network mapping states->features
    :param local_embedding: network mapping features to local embedding
    :param weight_out: weights that form the final layer of Q and reward prediction
    :param bias_out: bias for final layer, unused
    :param global_embedding: map from state/features -> successor features
    :param global_embedding_target: map state/features -> successor features from target network
    :param policy: policy defining distribution over next actions
    :param discount_factor: a discount factor in [0,1)
    :return:
    """
    reward_t, terminal_t = replay_tuple.reward_t.unsqueeze(-1), replay_tuple.terminal_t.unsqueeze(-1)
    state_features_t = featuriser(replay_tuple.state_t)

    local_embedding_t = _select(local_embedding(state_features=state_features_t), replay_tuple.action_t, 1)
    loss_rwd = functional.smooth_l1_loss(functional.linear(local_embedding_t, weight_out, bias_out), reward_t)

    with torch.no_grad():
        se_tp1 = global_embedding_target(state=replay_tuple.state_tp1)
        se_tp1_avg = (policy.expectation(se_tp1).unsqueeze(-1) * se_tp1).sum(1)

        sf_q_tp1 = functional.linear(se_tp1_avg, weight_out, bias_out)

    se_t = _select(global_embedding(state_features=state_features_t), replay_tuple.action_t, 1)
    q_t = functional.linear(se_t, weight_out, bias_out)

    loss_q = _td_loss(q_t, reward_t, sf_q_tp1, terminal_t, discount_factor)
    loss_sf = _td_loss(se_t, local_embedding_t, se_tp1_avg, terminal_t, discount_factor)

    return loss_sf, loss_q, loss_rwd
