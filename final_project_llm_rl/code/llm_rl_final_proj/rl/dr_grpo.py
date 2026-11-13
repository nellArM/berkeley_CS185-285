from __future__ import annotations

from typing import Tuple

import torch

from llm_rl_final_proj.models.logprobs import masked_mean
from llm_rl_final_proj.rl.clipped import ClippedSurrogateAlgorithm


class DrGRPO(ClippedSurrogateAlgorithm):
    """DrGRPO removes the GRPO sequence-length normalization and std scaling.

    1. Advantages are mean-centered group rewards without division by the group standard
       deviation (configured in online/train_rm_grpo.py).
    2. The clipped token objectives are *summed* over each response instead of averaged over
       its own length. The sum over the minibatch is divided by one shared constant (the
       number of completion tokens in the minibatch), so every token carries the same weight
       regardless of the length of the response it belongs to.
    """

    name = "dr_grpo"

    def surrogate(self, log_ratio: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low, high = self.clip_range()
        ratio = torch.exp(log_ratio)
        adv_tok = adv.unsqueeze(1)
        per_token = torch.minimum(ratio * adv_tok, ratio.clamp(low, high) * adv_tok)
        objective = masked_mean(per_token, mask)  # sum over all tokens / shared constant
        clipfrac = masked_mean(((ratio < low) | (ratio > high)).float(), mask)
        return objective, clipfrac
