"""
Script for running tests on output of run_atari.py.
Typical usage:
python3 /path/to/log_folder output_file.txt
Results will print to stdout.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as _mp

from atari_env import make_atari
from configs import construct_rl_modules
from models.policies import GreedyPolicy
from utilities import GPUCounter

mp = _mp.get_context('spawn')


EMULATOR_TIME_LIMIT = 30 * 60
EMULATOR_FPS = 60
N_REPEAT = 4


def emulator_time_exit(n_steps):
    return n_steps * N_REPEAT > EMULATOR_TIME_LIMIT * EMULATOR_FPS


class TestArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('test_folder', type=str)
        self.add_argument('output_file', type=str)
        self.add_argument('--n', type=int, default=100)
        self.add_argument('--n_process', type=int, default=4)
        self.add_argument('--n_gpu', type=int, default=-1)
        self.add_argument('--n_thread', type=int, default=2)
        self.add_argument('--max-checkpoint', type=float, default=0)
        self.add_argument('--no_time_limit', action='store_true',
                          help='Removes 30min emulator time limit.')


def play_test_episode(env, policy, emulator_time_limit=False):
    state, info = env.reset(), {'done': False}
    policy.start_new_episode()

    n_steps = 0
    while not info['done']:
        state, *_, info = env.step(policy(state.torch().cuda()))
        n_steps += 1

        if emulator_time_limit and emulator_time_exit(n_steps):
            break

    return info['ep_reward']


def set_seed():
    seed = int.from_bytes(os.urandom(4), sys.byteorder)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed & 0x00000000FFFFFFFF)


def run_test(input_args):
    test_args, (run_args, checkpoint_path), gpu_number = input_args
    set_seed()
    torch.set_num_threads(test_args.n_thread)

    if gpu_number >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)

    print(f"Running job on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set!')}")

    run_args = argparse.Namespace(**run_args)
    manager, env, model, _ = construct_rl_modules(run_args)
    state_dict = torch.load(checkpoint_path)[0]
    model.load_state_dict(state_dict)

    policy = GreedyPolicy(env.action_space, model.q_fn)

    env = make_atari(f'{run_args.game}NoFrameskip-v4')
    env._max_episode_steps = 999999999  # episode steps are controlled within the test loop.
    reward = np.mean([play_test_episode(env, policy, not test_args.no_time_limit) for _ in range(test_args.n)])

    return reward, checkpoint_path.split('/')[-1]


def test(path, test_args):
    with open(os.path.join(path, 'run_args.json'), 'r') as fp:
        checkpoint_args = json.load(fp)

    checkpoint_folder = os.path.join(path, 'checkpoints')
    checkpoints = os.listdir(checkpoint_folder)
    checkpoints = sorted(checkpoints, key=lambda x: float(x), reverse=True)
    if test_args.max_checkpoint:
        checkpoints = [c for c in checkpoints if float(c) / 10 <= test_args.max_checkpoint]
    max_reward = -float('inf')
    run_args = [(checkpoint_args, os.path.join(checkpoint_folder, str(checkpoint))) for checkpoint in checkpoints]

    rewards, n, total = [], 0, len(run_args)
    t0 = time.time()
    with mp.Pool(processes=test_args.n_process) as pool:
        gpus = [i % test_args.n_gpu for i in range(len(run_args))]
        for result in pool.imap_unordered(run_test, zip([test_args] * len(run_args), run_args, gpus)):
            reward, checkpoint = result
            rewards.append(reward)

            n += 1
            t1 = time.time()
            tps = (t1 - t0) / n
            print(f"Step {float(checkpoint)/10:.1f}M score {reward} --- done {n}/{total}, time per test {tps:.1f}")

            if reward > max_reward:
                max_reward = reward
                print(f'Max reward {max_reward} at checkpoint {checkpoint}')

    return max_reward


if __name__ == '__main__':
    args = TestArgParser().parse_args()
    counter = GPUCounter(args.n_gpu)

    try:
        os.remove(args.output_file)
    except FileNotFoundError:
        pass

    for directory_path, _, file_names in os.walk(args.test_folder):
        if 'run_args.json' not in file_names:
            continue

        score = test(directory_path, args)
        with open(args.output_file, 'a') as outfile:
            game, _, *settings = os.path.basename(directory_path).split('_')
            print(f'{game},{"_".join(settings[:-4])},{score}', file=outfile)
