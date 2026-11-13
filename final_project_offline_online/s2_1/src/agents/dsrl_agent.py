import copy
from typing import Sequence

import numpy as np
import torch
from torch import nn

import infrastructure.pytorch_util as ptu
from networks.rl_networks import LogParam


class DSRLAgent(nn.Module):
    """DSRL agent - https://arxiv.org/abs/2506.15799

    RL in the latent-noise space of a BC flow policy. Following the QAM variant, the BC flow
    policy is trained jointly (not frozen) and a target copy of it is used to decode noise.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_flow_actor,
        make_bc_flow_actor_optimizer,
        make_noise_actor,
        make_noise_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_z_critic,
        make_z_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        noise_scale: float = 1.0,
        online_training: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.target_entropy = -action_dim

        # BC flow actor and its target copy
        self.bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor = copy.deepcopy(self.bc_flow_actor)

        # Noise policy pi_z(z | s): tanh-squashed Gaussian over the latent noise
        self.noise_actor = make_noise_actor(observation_shape, action_dim)

        # Action-space critic Q(s, a) (+ target) and noise-space critic Q_z(s, z)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.z_critic = make_z_critic(observation_shape, action_dim)

        # Learnable entropy coefficient (stored as log alpha)
        self.log_alpha = LogParam(1.0)

        # Optimizers
        self.bc_flow_actor_optimizer = make_bc_flow_actor_optimizer(self.bc_flow_actor.parameters())
        self.noise_actor_optimizer = make_noise_actor_optimizer(self.noise_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.z_critic_optimizer = make_z_critic_optimizer(self.z_critic.parameters())
        self.alpha_optimizer = make_noise_actor_optimizer(self.log_alpha.parameters())

        self.to(ptu.device)

    @property
    def alpha(self):
        return self.log_alpha()

    @torch.compiler.disable
    def sample_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Euler integration of the (target) BC flow from t=0 to t=1, starting from `noises`."""
        actions = noises
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full((*actions.shape[:-1], 1), i * dt, device=actions.device)
            actions = actions + dt * self.target_bc_flow_actor(observations, actions, t)
        return torch.clamp(actions, -1, 1)

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions using noise policy for noise input to BC flow policy."""
        z = self.noise_actor(observations).sample()
        return self.sample_flow_actions(observations, self.noise_scale * z)

    def get_action(self, observation: np.ndarray):
        """Used for evaluation."""
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.sample_actions(observation)
        return ptu.to_numpy(action)[0]

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update critic: TD backup with a' decoded from the noise policy's latent."""
        with torch.no_grad():
            next_actions = self.sample_actions(next_observations)
            next_q = self.target_critic(next_observations, next_actions).mean(dim=0)
            target = rewards + self.discount * (1.0 - dones) * next_q

        q = self.critic(observations, actions)
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

    def update_qz(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        noises: torch.Tensor,
    ) -> dict:
        """Update z_critic: distill Q(s, pi_BC(s, z)) into Q_z(s, z) for z ~ N(0, I)."""
        with torch.no_grad():
            bc_actions = self.sample_flow_actions(observations, noises)
            target = self.critic(observations, bc_actions).mean(dim=0)

        qz = self.z_critic(observations, noises)
        loss = ((qz - target[None]) ** 2).mean(dim=1).sum()

        self.z_critic_optimizer.zero_grad()
        loss.backward()
        self.z_critic_optimizer.step()

        return {
            "qz_loss": loss,
            "qz_mean": qz.mean(),
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update BC flow actor with the flow-matching loss."""
        noise = torch.randn_like(actions)
        t = torch.rand((*actions.shape[:-1], 1), device=actions.device)
        interp = (1.0 - t) * noise + t * actions
        pred = self.bc_flow_actor(observations, interp, t)
        loss = ((pred - (actions - noise)) ** 2).mean()

        self.bc_flow_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_flow_actor_optimizer.step()

        return {
            "bc_flow_loss": loss,
        }

    def update_noise_actor(
        self,
        observations: torch.Tensor,
    ) -> dict:
        """Update noise actor (SAC-style primal step in the latent space)."""
        dist = self.noise_actor(observations)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        qz = self.z_critic(observations, self.noise_scale * z).mean(dim=0)
        loss = (self.alpha.detach() * log_prob - qz).mean()

        self.noise_actor_optimizer.zero_grad()
        loss.backward()
        self.noise_actor_optimizer.step()

        # Cache for the dual (alpha) update.
        self._last_log_prob = log_prob.detach()

        return {
            "noise_actor_loss": loss,
            "log_prob": log_prob.mean(),
            "z_abs_mean": z.abs().mean(),
        }

    def update_alpha(self) -> dict:
        """Update alpha (dual step): raise alpha when entropy is below the target."""
        loss = -(self.alpha * (self._last_log_prob + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()

        return {
            "alpha_loss": loss,
            "alpha": self.alpha,
        }

    def _polyak(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.mul_(1.0 - self.target_update_rate).add_(
                    param, alpha=self.target_update_rate
                )

    def update_target_critic(self) -> None:
        self._polyak(self.target_critic, self.critic)

    def update_target_bc_flow_actor(self) -> None:
        self._polyak(self.target_bc_flow_actor, self.bc_flow_actor)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        noises = torch.randn_like(actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_qz = self.update_qz(observations, actions, noises)
        metrics_actor = self.update_actor(observations, actions)
        metrics_noise_actor = self.update_noise_actor(observations)
        metrics_alpha = self.update_alpha()
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"z_critic/{k}": v.item() for k, v in metrics_qz.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"noise_actor/{k}": v.item() for k, v in metrics_noise_actor.items()},
            **{f"alpha/{k}": v.item() for k, v in metrics_alpha.items()},
        }

        self.update_target_critic()
        self.update_target_bc_flow_actor()

        return metrics
