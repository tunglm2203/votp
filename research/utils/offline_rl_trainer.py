from typing import Any, Callable, Dict, List, Optional, Union
from collections import defaultdict
import json
import os
import random
import time
import gym
import numpy as np
import torch
from tqdm import tqdm

from research.algs.base import Algorithm
from research.utils import evaluate
from research.utils.logger import Logger

MAX_VALID_METRICS = {"reward", "accuracy", "success", "is_success"}
LOG_LAST_METRICS = {"step", "steps"}


def log_from_dict(logger: Logger, metric_lists: Dict[str, Union[List, float]], prefix: str) -> None:
    keys_to_remove = []
    for metric_name, metric_value in metric_lists.items():
        if isinstance(metric_value, list) and len(metric_value) > 0:
            v = metric_value[-1] if metric_name in LOG_LAST_METRICS else np.mean(metric_value)
            logger.record(prefix + "/" + metric_name, v)
            keys_to_remove.append(metric_name)
        else:
            logger.record(prefix + "/" + metric_name, metric_value)
            keys_to_remove.append(metric_name)
    for key in keys_to_remove:
        del metric_lists[key]


def log_wrapper(fn: Callable, metric_lists: Dict[str, List]):
    def wrapped_fn(*args, **kwargs):
        metrics = fn(*args, **kwargs)
        for name, value in metrics.items():
            metric_lists[name].append(value)

    return wrapped_fn


def time_wrapper(fn: Callable, name: str, profile_lists: Dict[str, List]):
    def wrapped_fn(*args, timeit=False, **kwargs):
        if timeit:
            start_time = time.time()
            output = fn(*args, **kwargs)
            end_time = time.time()
            profile_lists[name].append(end_time - start_time)
        else:
            output = fn(*args, **kwargs)
        return output

    return wrapped_fn


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.utils.data.get_worker_info().seed
    seed = seed % (2**32 - 1)  # Reduce to valid 32bit unsigned range
    np.random.seed(seed)
    random.seed(seed)


def write_run_summary(path: str, eval_history: List[Dict[str, float]], summary_keys: List[str]) -> None:
    """Aggregate the eval history into end-of-training summary metrics and persist them.

    For each metric in ``summary_keys`` (higher-is-better), records the final eval value plus the
    best (max) and mean over the last 5 and 10 evals. Written to ``summary.json`` in the run dir and
    to ``wandb.run.summary`` (NOT through the CSV/TB logger, which would truncate log.csv on new keys).
    """
    if not eval_history:
        return
    summary = {}
    for key in summary_keys:
        vals = [h[key] for h in eval_history if key in h]
        if not vals:
            continue
        summary[f"final_{key}"] = float(vals[-1])
        for n in (5, 10):
            window = vals[-n:]
            summary[f"best{n}_{key}"] = float(np.max(window))
            summary[f"avg{n}_{key}"] = float(np.mean(window))
    with open(os.path.join(path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    try:
        import wandb
        if wandb.run is not None:
            wandb.run.summary.update({f"summary/{k}": v for k, v in summary.items()})
    except ImportError:
        pass


class OfflineRLTrainer(object):
    def __init__(
        self,
        eval_env: Optional[gym.Env] = None,
        total_steps: int = 1000,
        log_freq: int = 100,
        eval_freq: int = 1000,
        profile_freq: int = -1,
        checkpoint_freq: Optional[int] = None,
        max_validation_steps: Optional[int] = None,
        loss_metric: Optional[str] = "loss",
        x_axis: str = "steps",
        benchmark: bool = False,
        subproc_eval: bool = False,
        torch_compile: bool = False,
        torch_compile_kwargs: Dict = {},
        eval_fn: Optional[Any] = None,
        eval_kwargs: Dict = {},
        train_dataloader_kwargs: Dict = {},
        validation_dataloader_kwargs: Dict = {},
    ) -> None:
        self._model = None
        self.eval_env = eval_env

        # Logging parameters
        self.total_steps = total_steps
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.profile_freq = profile_freq
        self.checkpoint_freq = checkpoint_freq
        self.max_validation_steps = max_validation_steps
        self.loss_metric = loss_metric
        self.x_axis = x_axis

        # Performance parameters
        self.benchmark = benchmark
        assert subproc_eval == False, "Subproc eval not yet supported"
        assert torch_compile == False, "Torch Compile currently exhibits bugs. Do not use."
        self.torch_compile = torch_compile
        self.torch_compile_kwargs = torch_compile_kwargs

        # Eval parameters
        self.eval_fn = eval_fn
        self.eval_kwargs = eval_kwargs

        # Dataloader parameters
        self._train_dataloader = None
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self._validation_dataloader = None
        self.validation_dataloader_kwargs = validation_dataloader_kwargs
        self._validation_iterator = None

    def set_model(self, model: Algorithm):
        assert self._model is None, "Model has already been set."
        self._model = model

    @property
    def model(self):
        if self._model is None:
            raise ValueError("Model has not yet been set! use `set_model` before calling trainer functionality.")
        return self._model

    @property
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if not hasattr(self.model, "dataset"):
            self.model.setup_train_dataset()
        if self.model.dataset is None:
            return None
        if self._train_dataloader is None:
            shuffle = not isinstance(self.model.dataset, torch.utils.data.IterableDataset)
            pin_memory = self.model.device.type == "cuda"
            self._train_dataloader = torch.utils.data.DataLoader(
                self.model.dataset,
                shuffle=shuffle,
                pin_memory=pin_memory,
                worker_init_fn=_worker_init_fn,
                **self.train_dataloader_kwargs,
            )
        return self._train_dataloader

    @property
    def validation_dataloader(self) -> torch.utils.data.DataLoader:
        if not hasattr(self.model, "validation_dataset"):
            self.model.setup_validation_dataset()
        if self.model.validation_dataset is None:
            return None
        if self._validation_dataloader is None:
            kwargs = self.train_dataloader_kwargs.copy()
            kwargs.update(self.validation_dataloader_kwargs)
            shuffle = not isinstance(self.model.validation_dataset, torch.utils.data.IterableDataset)
            pin_memory = self.model.device.type == "cuda"
            self._validation_dataloader = torch.utils.data.DataLoader(
                self.model.validation_dataset,
                shuffle=shuffle,
                pin_memory=pin_memory,
                worker_init_fn=_worker_init_fn,
                **kwargs,
            )
        return self._validation_dataloader

    def check_compilation(self):
        # If the model has not been compiled, compile it.
        if not self.model.compiled and self.torch_compile:
            self.model.compile(**self.torch_compile_kwargs)

    def train(self, path: str):
        # Prepare the model for training by initializing the optimizers and the schedulers
        self.model.setup_optimizers()
        self.check_compilation()
        self.model.setup_schedulers()
        self.model.setup()  # perform any other arbitrary setup needs.
        print("[research] Training a model with", self.model.num_params, "trainable parameters.")

        # Setup benchmarking.
        if self.benchmark:
            torch.backends.cudnn.benchmark = True

        # Setup datasets
        self.model.setup_train_dataset()

        # Setup the Logger
        logger = Logger(path=path, writers=["tb", "csv", "wandb"])

        # Construct all the metric lists to be used during training
        train_metric_lists = defaultdict(list)
        profiling_metric_lists = defaultdict(list)
        # Wrap the functions we use in logging and profile wrappers
        train_step = log_wrapper(self.model.train_step, train_metric_lists)
        train_step = time_wrapper(train_step, "train_step", profiling_metric_lists)
        format_batch = time_wrapper(self.model.format_batch, "processor", profiling_metric_lists)

        # Compute validation trackers
        using_max_valid_metric = self.loss_metric in MAX_VALID_METRICS
        best_valid_metric = -1 * float("inf") if using_max_valid_metric else float("inf")

        profile = True if self.profile_freq > 0 else False  # must profile to get all keys for csv log

        env_domain = "metaworld" if "mw_" in self.eval_env.unwrapped.spec.name else "locomotion"
        # End-of-training summary is computed over the eval history for these metrics.
        summary_keys = ["success", "reward"] if env_domain == "metaworld" else ["score", "reward"]
        eval_history = []

        self.model.train()
        start_time = time.time()
        current_time = start_time
        pbar = tqdm(total=self.total_steps + 1, desc=f"Policy Training - {self.eval_env.unwrapped.spec.name}")
        step = 0
        while step <= self.total_steps:
            for batch in self.train_dataloader:
                if profile:
                    profiling_metric_lists["dataset"].append(time.time() - current_time)

                batch = format_batch(batch, timeit=profile)     # Next, format the batch
                train_step(batch, step, self.total_steps, timeit=profile)   # Run the train step

                # Update the schedulers
                for scheduler in self.model.schedulers.values():
                    scheduler.step()

                # Now determine if we should dump the logs
                if step % self.log_freq == 0:
                    # Record timing metrics
                    current_time = time.time()
                    logger.record("time/step", step)
                    logger.record("time/steps_per_second", self.log_freq / (current_time - start_time))

                    log_from_dict(logger, profiling_metric_lists, "time")
                    start_time = current_time
                    # Record learning rates
                    for name, scheduler in self.model.schedulers.items():
                        train_metric_lists["lr_" + name] = scheduler.get_last_lr()[0]
                    # Record training metrics
                    log_from_dict(logger, train_metric_lists, "train")
                    logger.dump(step=step)

                if step % self.eval_freq == 0 and not (step == 0 and self.benchmark):
                    self.model.eval()
                    current_valid_metric = None
                    model_metadata = dict(step=step)

                    # Run and time validation step
                    current_time = time.time()
                    validation_metrics = self.validate(path, step)
                    logger.record("time/validation", time.time() - current_time)
                    if self.loss_metric in validation_metrics:
                        current_valid_metric = validation_metrics[self.loss_metric]
                    log_from_dict(logger, validation_metrics, "validation")

                    # Run and time eval step
                    current_time = time.time()
                    eval_metrics = self.evaluate(path, step)
                    logger.record("time/eval", time.time() - current_time)
                    if self.loss_metric in eval_metrics:
                        current_valid_metric = eval_metrics[self.loss_metric]
                    if 'TimeLimit.truncated' in eval_metrics:  # Remove this to avoid overriding log.csv
                        eval_metrics.pop('TimeLimit.truncated')
                    # Capture before log_from_dict, which empties eval_metrics.
                    eval_history.append({k: eval_metrics[k] for k in summary_keys if k in eval_metrics})
                    log_from_dict(logger, eval_metrics, "eval")

                    # Determine if we have a new best self.model.
                    if current_valid_metric is None:
                        pass
                    elif (using_max_valid_metric and current_valid_metric >= best_valid_metric) or (
                        not using_max_valid_metric and current_valid_metric <= best_valid_metric
                    ):
                        best_valid_metric = current_valid_metric
                        self.model.save(path, "best_model", model_metadata)

                    logger.dump(step=step, eval=True)  # Eval Logger dump to CSV: Mark True on eval flag
                    self.model.save(path, "final_model", model_metadata)  # Save the final model every eval period
                    self.model.train()  # Put the model back in train mode

                # if self.checkpoint_freq is not None and (current_step - last_checkpoint) >= self.checkpoint_freq:
                if self.checkpoint_freq is not None and step % self.checkpoint_freq == 0:
                    # Save a checkpoint
                    model_metadata = dict(step=step)
                    self.model.save(path, "model_" + str(step), model_metadata)

                pbar.update(1)
                step += 1
                if step > self.total_steps:
                    break  # train_dataloader never exhausts (infinite sampler), so break out once we hit total_steps

                profile = self.profile_freq > 0 and step % self.profile_freq == 0
                if profile:
                    current_time = time.time()  # update current time only, not start time

        pbar.close()

        # Log end-of-training summary: final eval, best/mean over the last 5 and 10 evals.
        # Write to wandb.run.summary + summary.json, NOT through `logger` (prevent truncation in log.csv)
        write_run_summary(path, eval_history, summary_keys)

        logger.close()
        # Cleanup!
        model_metadata = dict(step=step)
        self.model.save(path, "final_model", model_metadata)
        if self.eval_env is not None:
            self.eval_env.close()

    def validate(self, path: str, step: int):
        assert not self.model.training
        return {}

    def evaluate(self, path: str, step: int):
        assert not self.model.training
        self.check_compilation()
        eval_fn = None if self.eval_fn is None else vars(evaluate)[self.eval_fn]
        if eval_fn is None:
            return dict()
        eval_metrics = eval_fn(self.eval_env, self.model, path, step, **self.eval_kwargs)
        return eval_metrics
