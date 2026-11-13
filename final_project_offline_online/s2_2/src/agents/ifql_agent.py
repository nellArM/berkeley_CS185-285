from typing import Sequence

import numpy as np
import torch
from torch import nn

import infrastructure.pytorch_util as ptu


class IFQLAgent(nn.Module):
    """Implicit flow Q-learning: IQL critics + BC flow actor + best-of-N (rejection) sampling."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        num_samples: int = 32,
        expectile: float = 0.9,
    ):
        super().__init__()

        self.action_dim = action_dim

        # Flow actor (BC vector field)
        self.actor_flow = make_actor_flow(observation_shape, action_dim)

        # Critic ensemble, target critic ensemble and value function
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)

        # Optimizers
        self.actor_flow_optimizer = make_actor_flow_optimizer(self.actor_flow.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL: l(x) = |tau - 1(x > 0)| x^2 with x = V(s) - Q(s, a).
        """
        weight = torch.abs(expectile - (adv > 0).float())
        return weight * adv ** 2

    @torch.compile
    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        with torch.no_grad():
            q = self.target_critic(observations, actions).min(dim=0).values
        v = self.value(observations)
        loss = self.expectile_loss(v - q, self.expectile).mean()

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "v_loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        batch_size = observations.shape[0]
        n = self.num_samples
        obs = observations.repeat_interleave(n, dim=0)  # (B * N, obs_dim)
        noise = torch.randn(batch_size * n, self.action_dim, device=observations.device)
        candidates = self.get_flow_action(obs, noise)  # (B * N, A), clipped to [-1, 1]
        values = self.critic(obs, candidates).min(dim=0).values.reshape(batch_size, n)
        best = values.argmax(dim=1)
        candidates = candidates.reshape(batch_size, n, self.action_dim)
        return candidates[torch.arange(batch_size, device=observations.device), best]

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.sample_actions(observation)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        action = noise
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full((*action.shape[:-1], 1), i * dt, device=action.device)
            action = action + dt * self.actor_flow(observation, action, t)
        return torch.clamp(action, -1, 1)

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        with torch.no_grad():
            next_v = self.value(next_observations)
            target = rewards + self.discount * (1.0 - dones) * next_v

        q = self.critic(observations, actions)  # (2, B)
        loss = ((q - target[None]) ** 2).mean(dim=1).sum()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        noise = torch.randn_like(actions)
        t = torch.rand((*actions.shape[:-1], 1), device=actions.device)
        interp = (1.0 - t) * noise + t * actions
        pred = self.actor_flow(observations, interp, t)
        loss = ((pred - (actions - noise)) ** 2).mean()

        self.actor_flow_optimizer.zero_grad()
        loss.backward()
        self.actor_flow_optimizer.step()

        return {
            "loss": loss,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_value(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.mul_(1.0 - self.target_update_rate).add_(
                    param, alpha=self.target_update_rate
                )
