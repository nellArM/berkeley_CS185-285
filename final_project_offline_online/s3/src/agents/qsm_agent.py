import math
from typing import Sequence

import numpy as np
import torch
from torch import nn

import infrastructure.pytorch_util as ptu


class QSMAgent(nn.Module):
    """Q-score matching with a DDPM policy (plus a DDPM behavior-cloning term, as in QAM's baseline)."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float,
        flow_steps: int,
        mean_target: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim

        # Actor: noise-prediction network eps(s, a_t, t)
        self.actor = make_actor(observation_shape, action_dim)

        # Critic ensemble and target
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps
        # False (default, as in the handout): clipped double-Q target, min over the two critics.
        # True: mean over the critics, as FQL uses (diagnostic for antsoccer, see report).
        self.mean_target = mean_target

        betas = self.cosine_beta_schedule(flow_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", torch.cumprod(alphas, dim=0))

        self.to(ptu.device)

    def cosine_beta_schedule(self, timesteps):
        """
        Cosine annealing beta schedule (Nichol & Dhariwal, 2021).
        """
        s = 0.008
        steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        f = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, 0.0, 0.999).float()

    def _predict_noise(self, observations: torch.Tensor, actions: torch.Tensor, t_index: torch.Tensor):
        """eps(s, a_t, t) with the integer timestep normalized to [0, 1)."""
        return self.actor(observations, actions, t_index.float() / self.flow_steps)

    @torch.compiler.disable
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor):
        """
        DDPM sampling: start from x_T = noise and denoise for `flow_steps` steps.
        """
        x = noise
        for i in reversed(range(self.flow_steps)):
            t = torch.full((*x.shape[:-1], 1), i, device=x.device)
            eps = self._predict_noise(observations, x, t)
            alpha, alpha_hat, beta = self.alphas[i], self.alpha_hats[i], self.betas[i]
            mean = (x - (1.0 - alpha) / torch.sqrt(1.0 - alpha_hat) * eps) / torch.sqrt(alpha)
            if i > 0:
                # Posterior variance beta_tilde = (1 - alpha_hat_{t-1}) / (1 - alpha_hat_t) * beta_t
                var = (1.0 - self.alpha_hats[i - 1]) / (1.0 - alpha_hat) * beta
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
            # Keep the iterate in a sane range; the first steps divide by a small sqrt(alpha).
            x = torch.clamp(x, -5.0, 5.0)
        return torch.clamp(x, -1, 1)

    def get_action(self, observation: torch.Tensor):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        with torch.no_grad():
            noise = torch.randn(observation.shape[0], self.action_dim, device=observation.device)
            action = self.ddpm_sampler(observation, noise)
        return ptu.to_numpy(action)[0]

    @torch.compiler.disable
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Critic
        """
        with torch.no_grad():
            next_actions = self.ddpm_sampler(next_observations, torch.randn_like(actions))
            next_qs = self.target_critic(next_observations, next_actions)
            next_q = next_qs.mean(dim=0) if self.mean_target else next_qs.min(dim=0).values
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

    @torch.compiler.disable
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor: Q-score matching term + alpha * DDPM (behavior cloning) term.
        """
        batch_size = actions.shape[0]
        noise = torch.randn_like(actions)
        t = torch.randint(0, self.flow_steps, (batch_size, 1), device=actions.device)
        alpha_hat = self.alpha_hats[t]  # (B, 1)
        noisy_actions = torch.sqrt(alpha_hat) * actions + torch.sqrt(1.0 - alpha_hat) * noise

        # Action gradient of Q at the noised action (no gradient into the critic's parameters).
        a = noisy_actions.detach().requires_grad_(True)
        q = self.critic(observations, a).mean(dim=0)
        q_grad = torch.autograd.grad(q.sum(), a)[0].detach()

        eps = self._predict_noise(observations, noisy_actions, t)

        # -eps plays the role of the (scaled) score; align it with eta * grad_a Q.
        qsm_loss = ((-eps - self.inv_temp * q_grad) ** 2).mean()
        ddpm_loss = ((noise - eps) ** 2).mean()
        loss = qsm_loss + self.alpha * ddpm_loss

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "total_loss": loss,
            "qsm_loss": qsm_loss,
            "ddpm_loss": ddpm_loss,
            "q_grad_abs_mean": q_grad.abs().mean(),
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
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
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
