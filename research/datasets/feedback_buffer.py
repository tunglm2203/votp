from typing import Dict, Optional, Union
import os
import io
import math
import gym
import numpy as np
import torch
from termcolor import cprint


import research
from research.utils import utils
from research.datasets.replay_buffer import get_buffer_bytes
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from research.rlhf_utils.mw_votp_labeling import mw_votp_labeling
from research.rlhf_utils.loco_votp_labeling import loco_votp_labeling


class PairwiseComparisonDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        path: Optional[str] = None,
        discount: float = 0.99,
        segment_size: int = 20,
        subsample_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
    ):
        super().__init__()
        self.discount = discount
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_size = segment_size
        self.subsample_size = subsample_size
        self._capacity = capacity

        if self._capacity is None:
            assert path is not None, "If capacity is not given, must have path to load from"
            with open(path, "rb") as f:
                data = np.load(f)
                data = utils.nest_dict(data)
            print(data["obs_1"].shape, observation_space)
            # Set the buffers to be the stored data. Woot woot.
            self.obs_1_buffer = utils.remove_float64(data["obs_1"])
            self.obs_2_buffer = utils.remove_float64(data["obs_2"])
            self.action_1_buffer = utils.remove_float64(data["action_1"])
            self.action_2_buffer = utils.remove_float64(data["action_2"])
            self.label_buffer = utils.remove_float64(data["label"])
            self._size = len(self.label_buffer)
        else:
            # Construct the buffers
            self.obs_1_buffer = utils.np_dataset_alloc(observation_space, self._capacity, begin_pad=(self.segment_size,))
            self.obs_2_buffer = utils.np_dataset_alloc(observation_space, self._capacity, begin_pad=(self.segment_size,))
            self.action_1_buffer = utils.np_dataset_alloc(action_space, self._capacity, begin_pad=(self.segment_size,))
            self.action_2_buffer = utils.np_dataset_alloc(action_space, self._capacity, begin_pad=(self.segment_size,))
            self.label_buffer = utils.np_dataset_alloc(0.5, self._capacity)
            self._size = 0
            self._idx = 0
            if path is not None:
                assert path is not None, "If capacity is not given, must have path to load from"
                with open(path, "rb") as f:
                    data = np.load(f)
                    data = {k: data[k] for k in data.keys()}
                    dataset_size = data["label"].shape[0]
                    if dataset_size > self._capacity:
                        # Trim the dataset down
                        data = utils.get_from_batch(data, 0, self._capacity)
                    data = utils.nest_dict(data)
                    self.add(data, data["label"])  # Add to the buffer via the add method!

        # Print the size of the allocation.
        storage = 0
        storage += 2 * get_buffer_bytes(self.obs_1_buffer)
        storage += 2 * get_buffer_bytes(self.action_1_buffer)
        storage += get_buffer_bytes(self.label_buffer)
        print("[PairwiseComparisonDataset] allocated {:.2f} GB".format(storage / 1024**3))

        # Clip everything
        lim = 1 - 1e-5
        self.action_1_buffer = np.clip(self.action_1_buffer, a_min=-lim, a_max=lim)
        self.action_2_buffer = np.clip(self.action_2_buffer, a_min=-lim, a_max=lim)

    def add(self, queries: Dict, labels: np.ndarray):
        assert self._capacity is not None, "Can only add to non-static buffers."
        assert torch.utils.data.get_worker_info() is None, "Cannot add to PairwiseComparisonDataset when parallelism is enabled."
        num_to_add = labels.shape[0]
        if self._idx + num_to_add > self._capacity:
            # We have more segments than capacity allows, complete in two writes.
            num_b4_wrap = self._capacity - self._idx
            self.add(utils.get_from_batch(queries, 0, num_b4_wrap), labels[:num_b4_wrap])
            self.add(utils.get_from_batch(queries, num_b4_wrap, num_to_add), labels[num_b4_wrap:])
        else:
            start, end = self._idx, self._idx + num_to_add
            utils.set_in_batch(self.obs_1_buffer, queries["obs_1"], start, end)
            utils.set_in_batch(self.obs_2_buffer, queries["obs_2"], start, end)
            utils.set_in_batch(self.action_1_buffer, queries["action_1"], start, end)
            utils.set_in_batch(self.action_2_buffer, queries["action_2"], start, end)
            self.label_buffer[start:end] = labels
            self._idx = (self._idx + num_to_add) % self._capacity
            self._size = min(self._size + num_to_add, self._capacity)

    def _sample(self, idxs):
        if self.subsample_size is None:
            obs_1 = utils.get_from_batch(self.obs_1_buffer, idxs)
            obs_2 = utils.get_from_batch(self.obs_2_buffer, idxs)
            action_1 = utils.get_from_batch(self.action_1_buffer, idxs)
            action_2 = utils.get_from_batch(self.action_2_buffer, idxs)
            label = self.label_buffer[idxs]
        else:
            # Note: subsample sequences currently do not support arbitrary obs/action spaces.
            # we could do this with two utils calls, but that would be slower.
            start = np.random.randint(0, self.segment_size - self.subsample_size)
            end = start + self.subsample_size
            obs_1 = self.obs_1_buffer[idxs, start:end]
            obs_2 = self.obs_2_buffer[idxs, start:end]
            action_1 = self.action_1_buffer[idxs, start:end]
            action_2 = self.action_2_buffer[idxs, start:end]
            label = self.label_buffer[idxs]

        batch_size = len(idxs)
        discount = self.discount * np.ones(batch_size, dtype=np.float32) if self.batch_size > 1 else self.discount
        return dict(obs_1=obs_1, obs_2=obs_2, action_1=action_1, action_2=action_2, label=label, discount=discount)

    def save(self, path):
        # Save everything to the path via savez
        data = dict(
            obs_1=utils.get_from_batch(self.obs_1_buffer, 0, self._size),
            obs_2=utils.get_from_batch(self.obs_2_buffer, 0, self._size),
            action_1=utils.get_from_batch(self.action_1_buffer, 0, self._size),
            action_2=utils.get_from_batch(self.action_2_buffer, 0, self._size),
            label=self.label_buffer[: self._size],
        )
        data = utils.flatten_dict(data)
        with io.BytesIO() as bs:
            np.savez_compressed(bs, **data)
            bs.seek(0)
            with open(path, "wb") as f:
                f.write(bs.read())

    def __len__(self):
        return self._size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        chunk_size = len(self) // num_workers
        my_inds = np.arange(chunk_size * worker_id, chunk_size * (worker_id + 1))
        idxs = np.random.permutation(my_inds)

        for i in range(math.ceil(len(idxs) / self.batch_size)):  # Need to use ceil to get all data points.
            if self.batch_size == 1:
                cur_idxs = idxs[i]
            else:
                # Might be some overlap here but its probably OK.
                cur_idxs = idxs[i * self.batch_size : min((i + 1) * self.batch_size, len(self))]
            yield self._sample(cur_idxs)


class D4RL_FeedbackDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            path: Optional[str] = None,
            discount: float = 0.99,
            segment_size: int = 20,
            subsample_size: Optional[int] = None,
            batch_size: Optional[int] = None,
            action_eps: float = 1e-5,
            equal_pref_threshold_teacher: float = 0.0,
            human_label: Optional[bool] = False,
            use_pseudo_label: Optional[bool] = False,
            preference_threshold: Optional[float] = 0.0,
            n_queries: Optional[int] = 10000,
            env_name: Optional[str] = None,
            entropic_reg: float = 0.01,
            n_labels: int = 10,
            max_pairs: int = 10000,
            seed: Optional[int] = -1,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.discount = discount
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_size = segment_size
        self.subsample_size = subsample_size


        # Variables for preference learning
        self.n_queries = n_queries  # this is only applicable when using ground-truth preference labels
        self.use_pseudo_label = use_pseudo_label
        self.preference_threshold = preference_threshold
        self.human_label = human_label
        assert n_queries <= max_pairs, "n_queries must be less than or equal to max_pairs"
        if self.human_label:
            assert equal_pref_threshold_teacher == 0, f"equal_pref_threshold_teacher is not applicable with human label"

        assert path is not None, "path to offline segment must be provided"
        # Load offline segments (we use 20k segments)
        with open(path, "rb") as f:
            data = np.load(f)
            data = utils.nest_dict(data)
        data = utils.remove_float64(data)
        lim = 1 - action_eps
        data["action_1"] = np.clip(data["action_1"], a_min=-lim, a_max=lim)
        data["action_2"] = np.clip(data["action_2"], a_min=-lim, a_max=lim)
        self.data = data

        # Generate VOTP pseudo preferences on the fly (teacher = human or script)
        teacher = "human" if self.human_label else "script"
        votp_out = loco_votp_labeling(
            env_name=env_name,
            entropic_reg=entropic_reg,
            preference_thresh=self.preference_threshold,
            equal_pref_threshold_teacher=equal_pref_threshold_teacher,
            n_labels=n_labels,
            max_pairs=max_pairs,
            teacher=teacher,
            return_training_data=True,
            use_gt_only=(self.use_pseudo_label == False),
        )

        if self.use_pseudo_label:
            # Get pseudo-labels above the preference threshold
            self.pair_indices = votp_out["retained_pair_indices"]
            self.preference_labels = votp_out["pseudo_labels"]
            cprint(f"[D4RL]: Use VOTP pseudo labels for unlabeled pairs\n", 'yellow', attrs=['bold'])
        else:
            # Ground-truth labels on the GT-available pairs (human teacher has a limited budget -> filter NaN)
            gt = votp_out["gt_preferences"]
            valid = ~np.isnan(gt)
            self.pair_indices = votp_out["pair_indices"][valid][:self.n_queries]
            self.preference_labels = gt[valid][:self.n_queries]
            cprint(f"[D4RL] Use ground-truth ({teacher}) labels for unlabeled pairs: n_pairs={len(self.preference_labels)}\n", 'yellow', attrs=['bold'])

        assert (np.unique(self.preference_labels) >= 0).all() and (np.unique(self.preference_labels) <= 1).all()

    def __len__(self):
        return len(self.preference_labels)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        chunk_size = len(self) // num_workers
        my_inds = np.arange(chunk_size * worker_id, chunk_size * (worker_id + 1))
        idxs = np.random.permutation(my_inds)

        for i in range(math.ceil(len(idxs) / self.batch_size)):  # Need to use ceil to get all data points.
            if self.batch_size == 1:
                cur_idxs = idxs[i]
                data_1_idxs, data_2_idxs = self.pair_indices[cur_idxs][0], self.pair_indices[cur_idxs][1]
                data_2_idxs = data_2_idxs - self.data["obs_1"].shape[0]  # This is specific for D4RL
            else:
                cur_idxs = idxs[i * self.batch_size: min((i + 1) * self.batch_size, len(self))]
                data_1_idxs, data_2_idxs = self.pair_indices[cur_idxs][:, 0], self.pair_indices[cur_idxs][:, 1]
                data_2_idxs = data_2_idxs - self.data["obs_1"].shape[0]  # This is specific for D4RL

            dataset_segment_length = self.data["action_1"].shape[1]
            if self.subsample_size is not None:
                start = np.random.randint(0, dataset_segment_length - self.subsample_size)
                start_1, end_1 = start, start + self.subsample_size
                start_2, end_2 = start, start + self.subsample_size
            else:
                start_1, end_1 = 0, dataset_segment_length
                start_2, end_2 = 0, dataset_segment_length

            batch = {
                "obs_1": self.data["obs_1"][data_1_idxs, start_1:end_1],
                "obs_2": self.data["obs_2"][data_2_idxs, start_2:end_2],
                "action_1": self.data["action_1"][data_1_idxs, start_1:end_1],
                "action_2": self.data["action_2"][data_2_idxs, start_2:end_2],
                "reward_1": self.data["reward_1"][data_1_idxs, start_1:end_1],
                "reward_2": self.data["reward_2"][data_2_idxs, start_2:end_2],
                "original_indices_1": self.data["original_indices_1"][data_1_idxs, start_1:end_1],
                "original_indices_2": self.data["original_indices_2"][data_2_idxs, start_2:end_2],
                "label": self.preference_labels[cur_idxs].astype(np.float32)
            }
            yield batch


class MW_FeedbackDataset(torch.utils.data.IterableDataset):
    """
    This modified PairwiseComparisonDataset is used to work with CPL dataset
    """
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            path: Optional[str] = None,
            discount: float = 0.99,
            subsample_size: Optional[int] = None,
            batch_size: Optional[int] = None,
            action_eps: float = 1e-5,
            equal_pref_threshold_teacher: float = 0.0,
            use_pseudo_label: Optional[bool] = False,
            preference_threshold: Optional[float] = 0.0,
            n_queries: Optional[int] = 50000,
            env_name: Optional[str] = None,
            entropic_reg: float = 0.01,
            n_labels: int = 10,
            max_pairs: int = 50000,
            seed: Optional[int] = -1,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.discount = discount
        self.batch_size = 1 if batch_size is None else batch_size
        self.subsample_size = subsample_size

        # Variables for preference learning
        self.n_queries = n_queries  # this is only used when using GT preference (i.e., use_pseudo_label=False)
        self.use_pseudo_label = use_pseudo_label
        self.preference_threshold = preference_threshold
        assert n_queries <= max_pairs, "n_queries must be less than or equal to max_pairs"

        assert path is not None, "path to offline segment must be provided"
        # Load offline segments (we use 20k segments from CPL)
        with open(path, "rb") as f:
            data = np.load(f)
            data = utils.nest_dict(data)
        data = utils.remove_float64(data)
        lim = 1 - action_eps
        data["action"] = np.clip(data["action"], a_min=-lim, a_max=lim)
        self.data = data

        # Generate VOTP pseudo preferences on the fly
        votp_out = mw_votp_labeling(
            env_name=env_name,
            entropic_reg=entropic_reg,
            preference_thresh=self.preference_threshold,
            equal_pref_threshold_teacher=equal_pref_threshold_teacher,
            n_labels=n_labels,
            max_pairs=max_pairs,
            return_training_data=True,
            use_gt_only=(self.use_pseudo_label == False)
        )

        if self.use_pseudo_label:
            # Get pseudo-labels above the preference threshold
            self.pair_indices = votp_out["retained_pair_indices"]
            self.preference_labels = votp_out["pseudo_labels"]
            cprint(f"[MetaWorld]: Use VOTP pseudo labels for unlabeled pairs\n", 'yellow', attrs=['bold'])
        else:
            # Ground-truth synthetic labels on the sampled pairs
            self.pair_indices = votp_out["pair_indices"][:self.n_queries]
            self.preference_labels = votp_out["gt_preferences"][:self.n_queries]
            cprint(f"[MetaWorld] Use ground-truth synthetic labels for unlabeled pairs: n_pairs={len(self.preference_labels)}\n", 'yellow', attrs=['bold'])

        assert (np.unique(self.preference_labels) >= 0).all() and (np.unique(self.preference_labels) <= 1).all()

    def __len__(self):
        return len(self.preference_labels)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        chunk_size = len(self) // num_workers
        my_inds = np.arange(chunk_size * worker_id, chunk_size * (worker_id + 1))
        idxs = np.random.permutation(my_inds)

        for i in range(math.ceil(len(idxs) / self.batch_size)):  # Need to use ceil to get all data points.
            if self.batch_size == 1:
                cur_idxs = idxs[i]
                data_1_idxs, data_2_idxs = self.pair_indices[cur_idxs][0], self.pair_indices[cur_idxs][1]
            else:
                cur_idxs = idxs[i * self.batch_size: min((i + 1) * self.batch_size, len(self))]
                data_1_idxs, data_2_idxs = self.pair_indices[cur_idxs][:, 0], self.pair_indices[cur_idxs][:, 1]

            dataset_segment_length = self.data["action"].shape[1]
            if self.subsample_size is not None:
                start = np.random.randint(0, dataset_segment_length - self.subsample_size)
                start_1, end_1 = start, start + self.subsample_size
                start_2, end_2 = start, start + self.subsample_size
            else:
                start_1, end_1 = 0, dataset_segment_length
                start_2, end_2 = 0, dataset_segment_length

            batch = {
                "obs_1": self.data["obs"][data_1_idxs, start_1:end_1],
                "obs_2": self.data["obs"][data_2_idxs, start_2:end_2],
                "action_1": self.data["action"][data_1_idxs, start_1:end_1],
                "action_2": self.data["action"][data_2_idxs, start_2:end_2],
                "reward_1": self.data["reward"][data_1_idxs, start_1:end_1],
                "reward_2": self.data["reward"][data_2_idxs, start_2:end_2],
                "original_indices_1": self.data["original_indices"][data_1_idxs, start_1:end_1],
                "original_indices_2": self.data["original_indices"][data_2_idxs, start_2:end_2],
                "label": self.preference_labels[cur_idxs].astype(np.float32)
            }
            yield batch


class ReplayAndFeedbackBuffer(torch.utils.data.IterableDataset):
    """
    Dataset class that combines a replay buffer and a feedback buffer
    This is used for IH Learn and Offline Learning
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        replay_class: Union[str, torch.utils.data.IterableDataset],
        feedback_class: Union[str, torch.utils.data.IterableDataset] = PairwiseComparisonDataset,
        replay_kwargs: Dict = {},
        feedback_kwargs: Dict = {},
        **kwargs,
    ):
        replay_kwargs = replay_kwargs.copy()
        replay_kwargs.update(kwargs)
        replay_class = vars(research.datasets)[replay_class] if isinstance(replay_class, str) else replay_class
        self.replay_buffer = replay_class(observation_space, action_space, **replay_kwargs)
        feedback_kwargs = feedback_kwargs.copy()
        feedback_kwargs.update(kwargs)
        feedback_class = vars(research.datasets)[feedback_class] if isinstance(feedback_class, str) else feedback_class
        self.feedback_dataset = feedback_class(observation_space, action_space, **feedback_kwargs)

    def __iter__(self):
        # Yield one batch of each in a tuple per step.
        replay_iter = iter(self.replay_buffer)
        feedback_iter = iter(self.feedback_dataset)
        current_feedback_size = len(self.feedback_dataset)

        while True:
            replay_data = next(replay_iter)  # Replay iter should be infinite
            if len(self.feedback_dataset) > current_feedback_size:
                # Check to see if the size of the feedback dataset has increased
                # If so, recreate the iterator to fetch new data.
                current_feedback_size = len(self.feedback_dataset)
                del feedback_iter
                feedback_iter = iter(self.feedback_dataset)

            feedback_data = next(feedback_iter, None)
            if feedback_data is None:
                # Check once to re-add. If this is the first epoch, we may get None back.
                feedback_iter = iter(self.feedback_dataset)
                feedback_data = next(feedback_iter, None)

            yield replay_data, feedback_data
