"""
Script to run experiments. See configs.py for command line argument parser. Usage:
python3 run_atari.py --game Breakout
will run the Successor Uncertainties model on Breakout with the parameters used in the paper.
Once run has finished, see run_test.py to obtain a test score.
"""

import logging
import time

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

import configs
from models.policies import UniformPolicy
from replay_buffer import ReplayBuffer


def play_episode(n_steps, manager, env, model, policy, replay_buffer, optim, config):
    ep_start_steps, t0 = n_steps, time.time()
    state, ep_reward, terminal = env.reset(), 0, False

    policy.start_new_episode()

    while not terminal:
        action = policy(state.torch().cuda())

        next_state, reward, terminal, info = env.step(action)

        replay_buffer.add(state, action, reward, next_state, terminal)
        policy.update(state, action, reward)
        state = next_state

        ep_reward += reward
        n_steps += 1

        if optim and n_steps % config.train_interval == 0:
            optim.zero_grad()

            if n_steps % config.update_interval == 0:
                model.update_params()

            model.loss(replay_buffer.sample()).backward()
            clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optim.step()

            manager.record_losses(n_steps, model.named_loss_dict)

    fps = (n_steps - ep_start_steps) / (time.time() - t0)
    manager.record_policy(n_steps, policy)
    manager.record_episode(n_steps, info, ep_reward, fps=fps)
    return n_steps


def run_atari_experiment(args, verbose=True, log_dir='logs'):
    config = configs.AtariArgParse().parse_args(args)
    manager, env, model, policy = configs.construct_rl_modules(config, log_dir)

    if not verbose:
        logging.getLogger().setLevel('ERROR')
    replay_buffer = ReplayBuffer(config.buffer_size, config.batch_size)

    n_steps = 0
    while n_steps < config.learning_start_step:
        n_steps = play_episode(n_steps, manager, env, model, UniformPolicy(policy.action_size),
                               replay_buffer, None, config)

    optim = Adam(model.parameters(), lr=config.lr)
    manager.training_started = True

    while n_steps < config.total_num_steps:
        n_steps = play_episode(n_steps, manager, env, model, policy, replay_buffer, optim, config)

    return manager.end_report()


if __name__ == '__main__':
    run_atari_experiment(args=None)
