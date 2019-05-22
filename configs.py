"""
Parser for command line arguments and helper function that converts these arguments into a set of RL modules required
to run experiments. Default values correspond to those used in the paper.
"""


import argparse

import models.linear
import models.models
import models.policies
from atari_env import make_atari
from utilities import AtariManager


class AtariArgParse(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--algorithm', choices=('epsgreedy', 'successor'), default='successor')

        self.add_argument('--game', type=str, default='Pong',
                          help='Atari game to use. Openai gym lists environments in the form XXXXNoFrameskip-v4; '
                               'provide the XXXX part here.')
        self.add_argument('--total_num_steps', type=int, default=int(1e8),
                          help='Total number of training steps to perform.')
        self.add_argument('--learning_start_step', type=int, default=int(1e5),
                          help='At the start of training learning_start_step number of steps are '
                               'executed using a uniform random policy, with no training taking place.')

        """Arguments relating to case algorithm=epsgreedy"""
        self.add_argument('--eps_end_value', type=float, default=0.01,
                          help='Epsilon in [0, 1] for training.')
        self.add_argument('--eps_end_step', type=int, default=int(1e6),
                          help='Number of steps for epsilon to decay from 1.0 to eps_end_value.')

        """Arguments relating to case algorithm=successor"""
        self.add_argument('--beta', type=float, default=1e-3, help='Noise variance for linear Bayesian model.')
        self.add_argument('--decay_factor', type=float, default=1e-5,
                          help='Forgetting parameter determining the the rate with which the influence of older '
                               'observations on the Linear Bayesian Model mean and covariance estimates decays.')
        self.add_argument('--resample_interval', type=int, default=250,
                          help='Number of steps after which a new Q function estimate is sampled for use with '
                               'Posterior Sampling. A new sample is always drawn at the start of a new episode.')
        # Todo: we never looked at the effect of this!
        self.add_argument('--successor_size', type=int, default=64,
                          help='Dimensionality of state-action embeddings '
                               'and thus of the successor features themselves.')

        self.add_argument('--buffer_size', type=int, default=int(1e6))
        self.add_argument('--batch_size', type=int, default=32)
        self.add_argument('--lr', type=float, default=5e-5)
        self.add_argument('--grad_clip_norm', type=float, default=10.0)

        self.add_argument('--update_interval', type=int, default=int(1e4))
        self.add_argument('--train_interval', type=int, default=4)

        self.add_argument('--hidden_size', type=int, default=1024)

        self.add_argument('--name', type=str, default='unnamed_test')


def construct_rl_modules(args, log_dir='logs'):
    manager = AtariManager(log_dir, f"{args.game}_{args.algorithm}_{args.name}")

    manager.register_args(args)
    env = make_atari(f'{args.game}NoFrameskip-v4')

    action_size = env.action_space.n

    if args.algorithm == 'epsgreedy':
        model = models.models.QNetwork(action_size, hidden_size=args.hidden_size).cuda()

        policy = models.policies.EpsGreedyPolicy(action_size, model.q_network.q_fn, args.eps_end_step,
                                                 args.eps_end_value)

    elif args.algorithm == 'successor':
        model = models.models.SFQNetwork(action_size, hidden_size=args.hidden_size,
                                         successor_size=args.successor_size).cuda()

        uncertainty_model = models.linear.SuccessorUncertaintyModel(
            input_size=args.successor_size, out_std=args.beta, bias=False,
            zero_mean_weights=True, decay_factor=args.decay_factor)

        policy = models.policies.UncertaintyPolicy(
            action_size=action_size, uncertainty_model=uncertainty_model, q_fn=model.q_fn,
            local_embedding=model.local_embedding, global_embedding=model.global_embedding,
            resample_every=args.resample_interval)
    else:
        raise ValueError(f"Algorithm {args.algorithm} not recognised.")

    manager.register_module(model)
    model.register_policy(policy)

    return manager, env, model, policy
