"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


def _build_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """MLP: [Linear -> ReLU] * len(hidden_dims) -> Linear."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.net = _build_mlp(state_dim, hidden_dims, chunk_size * action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.net(state)
        return out.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self(state)
        # Eq. (1): squared L2 norm over the whole chunk, averaged over the batch.
        return ((pred - action_chunk) ** 2).sum(dim=(1, 2)).mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        flat_dim = chunk_size * action_dim
        # Input: observation, flattened noisy chunk, flow timestep tau.
        self.net = _build_mlp(state_dim + flat_dim + 1, hidden_dims, flat_dim)

    def velocity(
        self, state: torch.Tensor, noisy_chunk: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        b = state.shape[0]
        x = torch.cat([state, noisy_chunk.reshape(b, -1), tau.reshape(b, 1)], dim=-1)
        return self.net(x).view(b, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        b = state.shape[0]
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(b, 1, 1, device=state.device, dtype=state.dtype)
        noisy = tau * action_chunk + (1.0 - tau) * noise
        target = action_chunk - noise
        pred = self.velocity(state, noisy, tau)
        # Eq. (2)
        return ((pred - target) ** 2).sum(dim=(1, 2)).mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        b = state.shape[0]
        a = torch.randn(
            b, self.chunk_size, self.action_dim, device=state.device, dtype=state.dtype
        )
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = torch.full((b, 1), i * dt, device=state.device, dtype=state.dtype)
            # Eq. (3): Euler integration from tau=0 to tau=1.
            a = a + dt * self.velocity(state, a, tau)
        return a


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
