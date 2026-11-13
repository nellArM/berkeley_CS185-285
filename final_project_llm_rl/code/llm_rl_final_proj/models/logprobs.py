from __future__ import annotations

import torch
import torch.nn.functional as F

from llm_rl_final_proj.models.load import PolicyModel

_NO_GRAD_ROW_CHUNK = 8


def compute_per_token_logprobs(
    model: PolicyModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Returns log p(x_t | x_<t) for t in [1, L-1]. Shape: [B, L-1]."""

    def _forward(ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        logits = out.logits[:, :-1, :]  # the logits at position t-1 score token t
        targets = ids[:, 1:]
        # Fused log_softmax + gather; cast to float32 so the normalizer is not computed in bf16.
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="none",
        )
        return (-nll).reshape(targets.shape)

    if enable_grad:
        return _forward(input_ids, attention_mask)

    # No-grad scoring is called on whole rollouts / eval batches, so process a few rows at a
    # time to bound the size of the dense [rows, L, vocab] logits tensor.
    pieces = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], _NO_GRAD_ROW_CHUNK):
            end = start + _NO_GRAD_ROW_CHUNK
            pieces.append(_forward(input_ids[start:end], attention_mask[start:end]))
    return torch.cat(pieces, dim=0)


def build_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    del pad_token_id
    # Entry t of the per-token log-prob tensor scores token t+1, so the first completion token
    # (token index `prompt_input_len`) lives at entry `prompt_input_len - 1`.
    seq_len = input_ids.shape[1]
    token_index = torch.arange(1, seq_len, device=input_ids.device).unsqueeze(0)
    is_completion = token_index >= int(prompt_input_len)
    not_pad = attention_mask[:, 1:].to(input_ids.device) > 0
    return (is_completion & not_pad).float()


def masked_sum(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions.

    Uses estimator: exp(delta) - delta - 1 where delta = log p_ref(a) - log p_new(a).
    """
    delta = (ref_logprobs - new_logprobs).clamp(-log_ratio_clip, log_ratio_clip)
    per_token = torch.exp(delta) - delta - 1.0
    return masked_mean(per_token, mask, eps=eps)
