from __future__ import annotations

from typing import Tuple

import torch

from llm_rl_final_proj.models.logprobs import masked_mean, masked_mean_per_row
from llm_rl_final_proj.rl.clipped import ClippedSurrogateAlgorithm


class GRPO(ClippedSurrogateAlgorithm):
    """GRPO update with a PPO-style clipped surrogate over completion tokens.

    Token-level ratios are clipped, the clipped objective is averaged over the completion
    tokens of each response (length normalization), and then averaged over responses.
    The training loop itself lives in `ClippedSurrogateAlgorithm.update`.
    """

    name = "grpo"

    def surrogate(self, log_ratio: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low, high = self.clip_range()
        ratio = torch.exp(log_ratio)
        adv_tok = adv.unsqueeze(1)
        per_token = torch.minimum(ratio * adv_tok, ratio.clamp(low, high) * adv_tok)
        seq_obj = masked_mean_per_row(per_token, mask)  # mean over the response's own tokens
        clipfrac = masked_mean(((ratio < low) | (ratio > high)).float(), mask)
        return seq_obj.mean(), clipfrac
