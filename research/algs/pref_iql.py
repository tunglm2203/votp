import itertools
from typing import Dict, Optional, Type

import numpy as np
import torch
from torch.nn import functional as F

from research.networks.base import ActorCriticValueRewardPolicy, ActorCriticValueRewardSpecialPolicy
from research.utils import utils
from research.algs.base import Algorithm


def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)


class PIQL(Algorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 1,
        expectile: Optional[float] = None,
        beta: float = 1,
        clip_score: float = 100.0,
        reward_steps: Optional[int] = None,
        reward_loss_type: str = "ce",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticValueRewardPolicy) or isinstance(self.network, ActorCriticValueRewardSpecialPolicy)
        self.tau = tau
        self.target_freq = target_freq
        self.expectile = expectile
        self.beta = beta
        self.clip_score = clip_score
        self.reward_steps = reward_steps
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]

        assert reward_loss_type in ["ce", "linear"]
        self.reward_loss_type = reward_loss_type
        utils.print_yellow(f"Debug: Reward loss type: {reward_loss_type}")

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def linear_BT_loss(self, pred_hat, label, segment_size):
        pred_hat = pred_hat + segment_size + 1e-5
        pred_prob = pred_hat / torch.sum(pred_hat, dim=1, keepdim=True)
        # label and pred_hat cross entropy loss
        loss = -torch.sum(label * torch.log(pred_prob), dim=1)
        return torch.sum(loss)

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        keys = ("actor", "critic", "value", "reward")
        default_kwargs = {}
        for key, value in self.optim_kwargs.items():
            if key not in keys:
                default_kwargs[key] = value
            else:
                assert isinstance(value, dict), "Special keys must be kwarg dicts"

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(self.optim_kwargs.get("actor", dict()))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = self.optim_class(actor_params, **actor_kwargs)

        # Update the encoder with the critic.
        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(self.optim_kwargs.get("critic", dict()))
        self.optim["critic"] = self.optim_class(self.network.critic.parameters(), **critic_kwargs)

        value_kwargs = default_kwargs.copy()
        value_kwargs.update(self.optim_kwargs.get("value", dict()))
        self.optim["value"] = self.optim_class(self.network.value.parameters(), **value_kwargs)

        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(self.optim_kwargs.get("reward", dict()))
        self.optim["reward"] = self.optim_class(self.network.reward.parameters(), **reward_kwargs)

    def setup(self):
        super().setup()
        self._data_pts_seen = 0
        self._flops = 0

    def train_reward_step(self, batch, env_domain=None):
        if env_domain == "metaworld":
            obs = torch.cat([batch["obs_1"][:, :-1, :], batch["obs_2"][:, :-1, :]], dim=0)
            action = torch.cat([batch["action_1"][:, 1:, :], batch["action_2"][:, 1:, :]], dim=0)  # (B, S+1)
        elif env_domain == "locomotion":
            obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)
            action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
        else:
            raise NotImplementedError

        self.network.reward.train()
        # Compute shapes and add everything to the batch dimension
        B, S = obs.shape[:2]
        flat_obs_shape = (B * S,) + obs.shape[2:]
        flat_action_shape = (B * S,) + action.shape[2:]
        obs = obs.view(flat_obs_shape)
        action = action.view(flat_action_shape)

        if self.network.reward.__class__.__name__ in ['DrQv2Reward', 'ContinuousMLPCritic']:
            reward = self.network.reward(obs, action)
        else:
            raise NotImplementedError

        # First update the reward net.
        E, B_times_S = reward.shape
        if self.reward_loss_type == "ce":
            assert B_times_S == B * S
            r1, r2 = torch.chunk(reward.view(E, B, S), 2, dim=1)  # Should return two (E, B, S)
            logits = r2.sum(dim=2) - r1.sum(dim=2)  # Sum across sequence dim, (E, B)
            labels = batch["label"].float().unsqueeze(0).expand(E, -1)  # Shape (E, B)
            assert labels.shape == logits.shape
            reward_loss = self.reward_criterion(logits, labels).mean()

            with torch.no_grad():
                prediction = (r2.sum(dim=-1, keepdim=True) > r1.sum(dim=-1, keepdim=True))
                ground_truth = torch.round(labels)
                if len(prediction.shape) == 3:
                    prediction = prediction.squeeze(2)
                if len(ground_truth.shape) == 3:
                    ground_truth = ground_truth.squeeze(2)
                assert len(prediction.shape) == 2 and prediction.shape == ground_truth.shape
                reward_accuracy = (prediction == ground_truth).float().mean()

        elif self.reward_loss_type == "linear":
            assert B_times_S == B * S
            member = 0
            r1, r2 = torch.chunk(reward.view(E, B, S), 2, dim=1)  # Should return two (E, B, S)
            pred_seg_sum_1 = torch.sum(r1[member], dim=1, keepdim=True)
            pred_seg_sum_2 = torch.sum(r2[member], dim=1, keepdim=True)
            pred_hat = torch.cat([pred_seg_sum_1, pred_seg_sum_2], dim=-1)

            labels = F.one_hot(batch["label"].long(), num_classes=2).float()
            reward_loss = self.linear_BT_loss(pred_hat, labels, segment_size=S) / labels.shape[0]
            with torch.no_grad():
                prediction = (r2.sum(dim=-1, keepdim=True) > r1.sum(dim=-1, keepdim=True))
                ground_truth = torch.round(batch["label"].float().unsqueeze(0).expand(E, -1))
                if len(prediction.shape) == 3:
                    prediction = prediction.squeeze(2)
                if len(ground_truth.shape) == 3:
                    ground_truth = ground_truth.squeeze(2)
                assert len(prediction.shape) == 2 and prediction.shape == ground_truth.shape
                reward_accuracy = (prediction == ground_truth).float().mean()
        else:
            raise ValueError

        self.optim["reward"].zero_grad(set_to_none=True)
        reward_loss.backward()
        self.optim["reward"].step()

        return dict(reward_loss=reward_loss.item(), reward=reward.mean().item(), reward_accuracy=reward_accuracy.item())

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        # replay_batch = batch
        replay_batch, _ = batch

        obs = replay_batch['obs']
        action = replay_batch["action"]
        next_obs = replay_batch['next_obs']
        discount = replay_batch["discount"]
        reward = replay_batch["reward"]

        # Apply Encoders
        obs = self.network.encoder(obs)
        with torch.no_grad():
            next_obs = self.network.encoder(next_obs)

        # compute the value loss
        with torch.no_grad():
            target_qs = self.target_network.critic(obs, action)
            target_q = torch.min(target_qs, dim=0)[0]
        vs = self.network.value(obs)
        v_loss = iql_loss(vs, target_q.expand(vs.shape[0], -1), self.expectile).mean()

        self.optim["value"].zero_grad(set_to_none=True)
        v_loss.backward()
        value_grad_norm = torch.nn.utils.clip_grad_norm_(self.network.value.parameters(), max_norm=1e9).item()
        self.optim["value"].step()

        # Next, update the actor. We detach and use the old value, v for computational efficiency
        # and use the target_q value though the JAX IQL recomputes both
        # Pytorch IQL versions have not.
        with torch.no_grad():
            adv = target_q - torch.mean(vs, dim=0)  # min trick is not used on value.
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        dist = self.network.actor(obs)  # Use encoder gradients for the actor.
        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(action).sum(dim=-1)
        elif torch.is_tensor(dist):
            assert dist.shape == action.shape
            bc_loss = torch.nn.functional.mse_loss(dist, action, reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")
        actor_loss = (exp_adv * bc_loss).mean()
        action_sat = (dist.abs() > 0.99).float().mean().item() if torch.is_tensor(dist) else None

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters()), max_norm=1e9).item()
        self.optim["actor"].step()

        # Next, Finally update the critic
        with torch.no_grad():
            next_vs = self.network.value(next_obs)
            next_v = torch.mean(next_vs, dim=0, keepdim=True)  # Min trick is not used on value.
            target = reward + discount * next_v  # use the predicted reward.
        qs = self.network.critic(obs, action)
        q_loss = torch.nn.functional.mse_loss(qs, target.expand(qs.shape[0], -1), reduction="none").mean()

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), max_norm=1e9).item()
        self.optim["critic"].step()

        metrics = dict(
            q_loss=q_loss.item(),
            v_loss=v_loss.item(),
            actor_loss=actor_loss.item(),
            v=vs.mean().item(),
            q=qs.mean().item(),
            q_max=qs.max().item(),
            q_min=qs.min().item(),
            q_std=qs.std().item(),
            td_error=(qs - target.expand_as(qs)).abs().mean().item(),
            adv=adv.mean().item(),
            adv_clip_frac=(exp_adv >= self.clip_score).float().mean().item() if self.clip_score is not None else 0.0,
            exp_adv_mean=exp_adv.mean().item(),
            exp_adv_max=exp_adv.max().item(),
            exp_adv_std=exp_adv.std().item(),
            bc_loss=bc_loss.mean().item(),
            grad_norm_actor=actor_grad_norm,
            grad_norm_critic=critic_grad_norm,
            grad_norm_value=value_grad_norm,
            reward=reward.mean().item(),
        )
        if action_sat is not None:
            metrics["action_sat"] = action_sat

        # Update the networks. These are done in a stack to support different grad options for the encoder.
        if step % self.target_freq == 0:
            with torch.no_grad():
                # Only run on the critic and encoder, those are the only weights we update.
                for param, target_param in zip(self.network.critic.parameters(), self.target_network.critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                # for param, target_param in zip(self.network.encoder.parameters(), self.target_network.encoder.parameters()):
                #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    def _predict(self, batch: Dict, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            z = self.network.encoder(batch["obs"])
            dist = self.network.actor(z)
            if isinstance(dist, torch.distributions.Distribution):
                action = dist.sample() if sample else dist.loc
            elif torch.is_tensor(dist):
                action = dist
            else:
                raise ValueError("Invalid policy output")
            action = action.clamp(*self.action_range)
        return action

    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=self._current_obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
