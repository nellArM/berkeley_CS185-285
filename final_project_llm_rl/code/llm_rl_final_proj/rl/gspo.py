from __future__ import annotations

from typing import Tuple

import torch

from llm_rl_final_proj.models.logprobs import masked_mean_per_row
from llm_rl_final_proj.rl.clipped import ClippedSurrogateAlgorithm


class GSPO(ClippedSurrogateAlgorithm):
    """Sequence-level clipped surrogate using geometric-mean likelihood ratios.

    The token log-ratios of a response are averaged into one sequence-level ratio
    s_i = exp(mean_t log(pi_theta / pi_old)) = (pi_theta(y_i | x) / pi_old(y_i | x))^(1 / |y_i|),
    and PPO clipping is applied once per response instead of once per token.
    """

    name = "gspo"

    def surrogate(self, log_ratio: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low, high = self.clip_range()
        seq_ratio = torch.exp(masked_mean_per_row(log_ratio, mask))  # [B]
        seq_obj = torch.minimum(seq_ratio * adv, seq_ratio.clamp(low, high) * adv)
        clipfrac = ((seq_ratio < low) | (seq_ratio > high)).float().mean()
        return seq_obj.mean(), clipfrac
