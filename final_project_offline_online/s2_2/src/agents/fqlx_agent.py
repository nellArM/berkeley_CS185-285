from typing import Sequence

import numpy as np
import torch

import infrastructure.pytorch_util as ptu
from agents.fql_agent import FQLAgent


class FQLXAgent(FQLAgent):
    """FQL-X: flow Q-learning with critic-guided action selection for offline-to-online RL.

    The training objectives are those of FQL. On top of them:
      * Actions (for both evaluation and online data collection) are chosen by best-of-N
        sampling: draw N candidates from the one-step policy and execute the one with the
        highest pessimistic ensemble value  mean_i Q_i - rho * std_i Q_i.
      * The critic may be an ensemble of more than two members (set in the config).
      * The behavior-regularization strength can be changed for the online phase
        (`online_alpha`), since the constraint to the offline data matters less once the
        agent can correct its own value estimates with fresh experience.
    The replay strategy (symmetric offline/online sampling) and the update-to-data ratio are
    handled by the training script (`--offline_ratio`, `--utd`).
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        *args,
        num_samples: int = 8,
        rho: float = 0.0,
        online_alpha: float = -1.0,
        **kwargs,
    ):
        super().__init__(observation_shape, action_dim, *args, **kwargs)
        self.num_samples = int(num_samples)
        self.rho = rho
        self.online_alpha = online_alpha

    def set_online(self, online: bool) -> None:
        if online and self.online_alpha is not None and self.online_alpha >= 0:
            self.alpha = self.online_alpha

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        n = self.num_samples
        if n <= 1:
            noise = torch.randn(batch_size, self.action_dim, device=observations.device)
            return torch.clamp(self.onestep_actor(observations, noise), -1, 1)
        obs = observations.repeat_interleave(n, dim=0)
        noise = torch.randn(batch_size * n, self.action_dim, device=observations.device)
        candidates = torch.clamp(self.onestep_actor(obs, noise), -1, 1)
        qs = self.critic(obs, candidates)  # (ensemble, B * N)
        values = qs.mean(dim=0)
        if self.rho > 0:
            values = values - self.rho * qs.std(dim=0)
        best = values.reshape(batch_size, n).argmax(dim=1)
        candidates = candidates.reshape(batch_size, n, self.action_dim)
        return candidates[torch.arange(batch_size, device=observations.device), best]

    def get_action(self, observation: np.ndarray):
        observation = ptu.from_numpy(np.asarray(observation))[None]
        return ptu.to_numpy(self.sample_actions(observation))[0]
