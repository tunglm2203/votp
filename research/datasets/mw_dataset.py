from typing import Optional, Union, Callable, Dict

import os
import copy
import random
import gym
import numpy as np
import functools
import tempfile

import torch

from research.utils import utils
from . import storage, sampling


class MW_OfflineDataset(torch.utils.data.IterableDataset):
    """
    This class is designed to be able to produce the same dataset configs used in the IQL paper.
    See https://github.com/ikostrikov/implicit_q_learning
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        sample_fn: Union[str, Callable] = "sample",
        sample_kwargs: Optional[Dict] = None,
        path: Optional[str] = None,
        action_eps: float = 0.00001,
        capacity: Optional[int] = None,
        steps_after_success: Optional[int] = 1,
        distributed: bool = False,
        fetch_every: int = 1000,
        cleanup: bool = True,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_eps = action_eps
        self.steps_after_success = steps_after_success

        self.exclude_keys = []

        buffer_space = {
            "obs": self.observation_space,
            "action": self.action_space,
            "reward": 0.0,
            "done": False,
            "discount": 1.0,
        }
        flattened_buffer_space = utils.flatten_dict(buffer_space)
        self.buffer_space = utils.nest_dict(flattened_buffer_space)

        self.capacity = capacity

        # Setup the sampler
        if isinstance(sample_fn, str):
            sample_fn = vars(sampling)[sample_fn]

        # Use functools partial to override the default args.
        sample_kwargs = {} if sample_kwargs is None else sample_kwargs
        self.sample_fn = functools.partial(sample_fn, **sample_kwargs)

        # Path for preloaded data
        self.path = path

        # Setup based on distributed value
        self.distributed = distributed

        if self.distributed:
            self.cleanup = cleanup
            self.fetch_every = fetch_every
            if self.capacity is not None:
                self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")
                print("[research] Replay Buffer Storage Path", self.storage_path)
                self.current_ep = utils.nest_dict({k: list() for k in flattened_buffer_space.keys()})
            self.num_episodes = 0
        else:
            self._alloc(self.capacity)  # Alloc immediately

    def _alloc(self, capacity):
        # Create the data generator
        self._current_data_generator = self._data_generator()
        assert capacity is None
        # Allocte the entire dataset
        data = utils.concatenate(*list(self._current_data_generator), dim=0)
        ep_rewards, ep_lengths = [], []
        ep_reward, ep_length = 0, 0
        for i in range(data['obs'].shape[0]):
            ep_reward += data['reward'][i]
            ep_length += 1
            if data['done'][i]:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward, ep_length = 0, 0
        min_return, max_return, avg_return = np.min(ep_rewards), np.max(ep_rewards), np.mean(ep_rewards)
        min_len, max_len, avg_len = np.min(ep_lengths), np.max(ep_lengths), np.mean(ep_lengths)
        print(f"[research] MetaWorld return range: min={min_return:.2f}, max={max_return:.2f}, mean={avg_return:.2f}. Length: min={min_len}, max={max_len}, mean={avg_len:.2f}")
        if self.action_eps > 0.0:
            action_ = np.clip(data['action'], -1.0 + self.action_eps, 1.0 - self.action_eps)
            data['action'] = action_

        # ===== Override done following success flag =====
        _new_done = [False]
        success_cnt = 0
        for i in range(1, data['obs'].shape[0]):
            if data['success'][i]:
                success_cnt += 1

            if data['done'][i - 1] and (not data['done'][i]): # start new episode
                success_cnt = 0

            if success_cnt >= self.steps_after_success or data['done'][i]:
                _new_done.append(True)
            else:
                _new_done.append(False)

        _new_done = np.array(_new_done)
        data['terminal'] = copy.deepcopy(data['done'])
        data['done'] = _new_done
        # ===== Complete override

        self._storage = storage.FixedStorage(data)
        print("[ReplayBuffer] Allocated {:.2f} GB".format(self._storage.bytes / 1024 ** 3))

    def _data_generator(self):
        """
        Can be overridden in order to load the initial data differently.
        By default assumes the data to be the standard format, and returned as a data dictionary.
        or
        None

        This function can be overriden by sub-classes in order to produce data batches.
        It should do the following:
        1. split data across torch data workers
        2. randomize the order of data
        3. yield data of the form dicts
        """
        if self.path is None:
            return

        # By default get all of the file names that are distributed at the correct index
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        ep_filenames = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(".npz")]
        random.shuffle(ep_filenames)  # Shuffle all the filenames

        if num_workers > 1 and len(ep_filenames) == 1:
            print(
                "[ReplayBuffer] Warning: using multiple workers but single replay file. Reduce memory usage by sharding"
                " data with `save` instead of `save_flat`."
            )
        elif num_workers > 1 and len(ep_filenames) < num_workers:
            print("[ReplayBuffer] Warning: using more workers than dataset files.")

        for ep_filename in ep_filenames:
            ep_idx, _ = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            # Spread loaded data across workers if we have multiple workers and files.
            if ep_idx % num_workers != worker_id and len(ep_filenames) > 1:
                continue  # Only yield the files belonging to this worker.
            data = storage.load_data(ep_filename, exclude_keys=self.exclude_keys)
            yield data

    def __iter__(self):
        assert not hasattr(self, "_iterated"), "__iter__ called twice!"
        self._iterated = True
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is not None) == self.distributed, "ReplayBuffer.distributed not set correctly!"

        # allocate the buffer with the given capacity
        if self.distributed:
            self._alloc(None if self.capacity is None else self.capacity // worker_info.num_workers)
            self._episode_filenames = set()

        self._learning_online = False

        batch_size = self.sample_fn.keywords.get("batch_size", 1)
        stack_size = self.sample_fn.keywords.get("stack", 1)
        seq_size = self.sample_fn.keywords.get("seq_length", 1)

        while True:
            if self._storage.size < seq_size * stack_size + 1:
                yield {}  # If the buffer is too small for sampling, continue.
            else:
                sample = self.sample_fn(self._storage)
                if batch_size == 1:
                    sample = utils.squeeze(sample, 0)
                yield sample

    def __del__(self):
        if not self.distributed:
            return
        if self.cleanup:
            return
        else:
            paths = [os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path)]
            for path in paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.rmdir(self.storage_path)
            except OSError:
                pass