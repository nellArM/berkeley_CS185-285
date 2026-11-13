from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from llm_rl_final_proj.models.load import PolicyModel
from llm_rl_final_proj.models.logprobs import (
    approx_kl_from_logprobs,
    compute_per_token_logprobs,
    masked_mean,
)
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch, iter_minibatches
from llm_rl_final_proj.utils.torch_utils import clip_grad_norm_


class ClippedSurrogateAlgorithm(RLAlgorithm):
    """Shared PPO-style training loop for the GRPO family.

    Subclasses only define `surrogate`, i.e. how token log-ratios and the scalar per-response
    advantage are turned into the (to be maximized) objective. Everything else is identical:
    PPO epochs over shuffled minibatches, the sampled-token KL penalty towards the frozen
    reference, gradient accumulation, gradient clipping and logging.
    """

    def clip_range(self) -> Tuple[float, float]:
        cfg = self.cfg
        high = cfg.clip_eps_high if cfg.clip_eps_high > 0.0 else cfg.clip_eps
        return 1.0 - cfg.clip_eps, 1.0 + high

    def surrogate(
        self,
        log_ratio: torch.Tensor,  # [B, L-1], log pi_theta - log pi_old at sampled tokens (clamped)
        adv: torch.Tensor,        # [B]
        mask: torch.Tensor,       # [B, L-1] float completion mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (scalar objective to maximize, fraction of clipped ratios for logging)."""
        raise NotImplementedError

    def update(
        self,
        model: PolicyModel,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        cfg = self.cfg
        model.train()
        model.config.use_cache = False

        total_loss = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        total_entropy = 0.0
        n_mb = 0
        accum = 0
        skipped_empty = 0
        skipped_nonfinite = 0
        total_grad_norm = 0.0
        opt_steps = 0
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        accum_steps = max(1, grad_accum_steps)

        optimizer.zero_grad(set_to_none=True)
        rng = torch.Generator(device=rollout.input_ids.device)
        rng.manual_seed(self._next_update_seed())

        for _ in range(cfg.ppo_epochs):
            for mb in iter_minibatches(
                rollout,
                cfg.minibatch_size,
                shuffle=True,
                generator=rng,
                device=next(model.parameters()).device,
            ):
                adv = mb.advantages.clamp(-cfg.adv_clip, cfg.adv_clip).detach()
                mask = mb.completion_mask
                if float(mask.sum().item()) <= 0.0:
                    skipped_empty += 1
                    continue

                new_logp = compute_per_token_logprobs(model, mb.input_ids, mb.attention_mask)
                log_ratio = (new_logp - mb.old_logprobs).clamp(-20.0, 20.0)
                objective, clipfrac = self.surrogate(log_ratio, adv, mask)
                pg_loss = -objective
                kl = approx_kl_from_logprobs(new_logp, mb.ref_logprobs, mask)
                entropy = -masked_mean(new_logp.detach(), mask)  # logging only

                loss = (pg_loss + cfg.kl_coef * kl) / accum_steps
                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    continue
                loss.backward()
                accum += 1

                if (accum % accum_steps) == 0:
                    gnorm = clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                    if not math.isfinite(gnorm):
                        skipped_nonfinite += 1
                        optimizer.zero_grad(set_to_none=True)
                        accum = 0
                        continue
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    total_grad_norm += float(gnorm)
                    opt_steps += 1

                total_loss += float((loss.detach() * accum_steps).item())
                total_kl += float(kl.detach().item())
                total_entropy += float(entropy.item())
                total_clipfrac += float(clipfrac.detach().item())
                n_mb += 1

        if accum > 0 and (accum % accum_steps) != 0:
            gnorm = clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            if math.isfinite(gnorm):
                optimizer.step()
                total_grad_norm += float(gnorm)
                opt_steps += 1
            else:
                skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)

        denom = max(1, n_mb)
        return {
            "train/policy_loss_with_kl_penalty_mean_over_minibatches": total_loss / denom,
            "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches": total_kl / denom,
            "train/policy_token_entropy_mean_over_minibatches": total_entropy / denom,
            "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches": total_clipfrac / denom,
            "train/count_minibatches_skipped_because_completion_mask_had_no_tokens": float(skipped_empty),
            "train/count_update_attempts_skipped_due_to_nonfinite_loss_or_gradients": float(skipped_nonfinite),
            "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps": total_grad_norm / max(1, opt_steps),
            "train/count_optimizer_steps_per_training_iteration": float(opt_steps),
        }
