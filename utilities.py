"""
Utilities that handle multi-gpu testing, logging stats to tensorboard and saving checkpoints.
"""

import json
import logging
import os
import shutil
import sys
import time
import uuid

import tensorboardX
import torch
from collections import deque

import numpy as np
import multiprocessing as mp


def _get_rand_str(length=6):
    return uuid.uuid4().hex[-length:]


class GPUCounter:
    def __init__(self, max_value=1):
        self.val = mp.Value('i', 0)
        self.lock = mp.Lock()
        self._max_value = max_value

    def get_val(self):
        if self._max_value < 0:
            return -1

        with self.lock:
            val = self.val.value
            self.val.value = (val + 1) % self._max_value
            return val


class SmoothingContainer:
    def __init__(self, smoothing_level):
        self._smoothers = {}
        self.smoothing_level = smoothing_level

    def append(self, key, value):
        if key not in self._smoothers:
            self._smoothers[key] = deque(maxlen=self.smoothing_level)

        self._smoothers[key].append(value)

    def get(self, key):
        if len(self._smoothers[key]) == self.smoothing_level:
            return np.mean(self._smoothers[key])
        else:
            return None


class TrainingManager:
    def __init__(self, log_dir, tag, leave_no_trace=False):
        self.leave_no_trace = leave_no_trace
        if leave_no_trace:
            log_dir = '/tmp'

        self.log_path = TrainingManager.make_log_path(log_dir, tag)
        os.makedirs(self.log_path, exist_ok=True)

        self.checkpoint_base_path = os.path.join(self.log_path, 'checkpoints')
        os.makedirs(self.checkpoint_base_path, exist_ok=True)
        self.logger = self.setup_logging(self.log_path)
        self.tensorboard = tensorboardX.SummaryWriter(logdir=self.log_path)

        self._registered_modules = []

    def __del__(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

        if self.leave_no_trace:
            self.delete_log()

    def register_args(self, args):
        if not isinstance(args, dict):
            args = vars(args)
        arg_file_path = os.path.join(self.log_path, "run_args.json")
        with open(arg_file_path, 'w') as file_handle:
            json.dump(args, file_handle)
        for arg, value in args.items():
            self.tensorboard.add_text(arg, str(value))
        return self

    def register_module(self, m):
        self._registered_modules.append(m)

    def register_modules(self, ms):
        for m in ms:
            self.register_module(m)

    def checkpoint(self, step):
        save_dict = {i: m.state_dict() for i, m in
                     enumerate(self._registered_modules)}
        torch.save(save_dict, self.checkpoint_path(step))

    def restore(self, step):
        save_dict = torch.load(self.checkpoint_path(step))
        for i, m in enumerate(self._registered_modules):
            m.load_state_dict(save_dict[i])

    def checkpoint_path(self, step):
        return os.path.join(self.checkpoint_base_path, str(step))

    def delete_log(self):
        shutil.rmtree(self.log_path)

    @staticmethod
    def make_log_path(log_dir, tag):
        time_str = time.strftime("%d-%H_%M_%S", time.localtime())
        rand_str = _get_rand_str(6)
        file_name = f"{tag}_{time_str}_{rand_str}" \
            if tag is not None else f"{time_str}_{rand_str}"
        return os.path.join(log_dir, file_name)

    @staticmethod
    def setup_logging(log_path):
        logger = logging.getLogger()
        logger.propagate = False
        logger.handlers = []
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s::%(message)s')

        handler = logging.FileHandler(os.path.join(log_path, 'out.log'))
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger


class AtariManager(TrainingManager):
    def __init__(self, log_dir, tag, leave_no_trace=False):
        super().__init__(log_dir, tag, leave_no_trace)

        self._smoothed_losses = SmoothingContainer(1000)
        self._smoothed_ep_info = SmoothingContainer(100)

        self._last_100_real_rewards = deque(maxlen=100)
        self._max_score = None

        self._last_ep_step = self._last_export_step = 0
        self._covariance_log_interval = 1000  # 25000

        self.training_started = False

        self.handle = None

    def record_episode(self, n_steps, info, ep_reward, fps):
        values = {'steps_per_episode': n_steps - self._last_ep_step,
                  'reward': ep_reward}
        self._last_ep_step = n_steps

        if self.training_started:
            values['fps'] = fps

        if info['done']:
            values['real_reward'] = info['ep_reward']
            self._last_100_real_rewards.append(info['ep_reward'])
            if len(self._last_100_real_rewards) == 100:
                real_rewards_mean = np.mean(self._last_100_real_rewards)
                if self._max_score is None or real_rewards_mean > self._max_score:
                    self._max_score = real_rewards_mean
                    self.tensorboard.add_scalar('test/max_score', self._max_score, n_steps)

        self._record_smoothed(n_steps, values, 'training', self._smoothed_ep_info)

        self.logger.log(logging.INFO, msg=f"\r{n_steps / 1000:.0f}k steps :: {fps:.1f} fps, reward {ep_reward}")

        if n_steps - self._last_export_step > int(5e5):
            self.checkpoint(f'{np.round(n_steps / 1e5)}')
            self._last_export_step = n_steps

    def _record_smoothed(self, n_steps, raw_dict, category, smoothing_container):
        # plot raw values
        for loss_name, loss_value in raw_dict.items():
            self.tensorboard.add_scalar(f'{category}/{loss_name}', loss_value, n_steps)
            smoothing_container.append(loss_name, loss_value)

        # plot smoothed values
        for loss_name in raw_dict.keys():
            smoothed_val = smoothing_container.get(loss_name)
            if smoothed_val is not None:
                self.tensorboard.add_scalar(f'{category}_smoothed/{loss_name}', smoothed_val, n_steps)

    def record_losses(self, n_steps, named_loss_dict):
        self._record_smoothed(n_steps, named_loss_dict, 'loss', self._smoothed_losses)

        if len(named_loss_dict.keys()) > 2:
            # plot loss ratios
            smoothed_total = self._smoothed_losses.get('total')
            if smoothed_total is not None and smoothed_total != 0:
                for loss_name in named_loss_dict.keys():
                    if loss_name != 'total':
                        loss_value_smoothed = self._smoothed_losses.get(loss_name)
                        self.tensorboard.add_scalar(f'loss_ratio/{loss_name}',
                                                    loss_value_smoothed/smoothed_total, n_steps)

    def record_policy(self, n_steps, policy):
        p_stats, avg_q_value, action_history = policy.get_stats()
        covar = policy.get_covariance()

        if covar is not None:
            diagonal = covar.diag()
            self.tensorboard.add_scalar('covar/min_diag', diagonal.min().item(), n_steps)
            self.tensorboard.add_scalar('covar/max_diag', diagonal.max().item(), n_steps)
            self.tensorboard.add_scalar('covar/mean_diag', diagonal.mean().item(), n_steps)

        if action_history:
            self.tensorboard.add_histogram('training/action_history',
                                           torch.as_tensor(action_history, dtype=torch.uint8),
                                           n_steps)

        for name, value in p_stats.items():
            self.tensorboard.add_scalar(f'p_stats/{name}', value, n_steps)

        self.tensorboard.add_scalar('training/avg_q_value', avg_q_value, n_steps)

    def end_report(self):
        return np.mean(self._last_100_real_rewards)
