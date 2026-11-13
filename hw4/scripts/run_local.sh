#!/usr/bin/env bash
# Local (non-Modal) launcher for the four required HW4 runs and the GRPO ablations.
# The hyperparameters are identical to the handout commands; only --output_dir is a local
# path instead of /vol/runs/..., and wandb runs in offline mode.
#
# Usage: scripts/run_local.sh <run_name> [gpu_id]
#   run_name: format_copy_grpo | format_copy_reinforce | math_hard_reinforce | math_hard_grpo
#             | abl_<tag>   (see the case statement below)
set -euo pipefail
NAME="$1"
export CUDA_VISIBLE_DEVICES="${2:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

COMMON=(--batch_size 8 --lr 3e-5 --minibatch_size 8 --kl_coef 0.05 --max_grad_norm 0.5
        --wandb_enabled --wandb_project llm-rl-hw4 --sample_markdown_log_interval 1 --sample_log_interval 10)
FC=(--task format_copy --steps 51 --group_size 6 --min_new_tokens 1 --max_new_tokens 24
    --sample_log_n 6 --eval_interval 50 --save_interval 50 --warmup_steps 10)
MH=(--task math_hard --group_size 8 --min_new_tokens 8 --max_new_tokens 512 --max_prompt_tokens 512
    --temperature 0.8 --top_p 0.95 --grad_accum_steps 8 --sample_log_n 8
    --cuda_empty_cache_interval 50 --eval_interval 100 --save_interval 100)

run() { uv run python -m hw4.train --output_dir "runs/$NAME" --wandb_name "$NAME" "$@"; }

case "$NAME" in
  format_copy_grpo)      run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 2 --grad_accum_steps 6 --clip_eps 0.2 ;;
  format_copy_reinforce) run "${COMMON[@]}" "${FC[@]}" --algo reinforce --grad_accum_steps 6 ;;
  math_hard_reinforce)   run "${COMMON[@]}" "${MH[@]}" --algo reinforce --steps 201 ;;
  math_hard_grpo)        run "${COMMON[@]}" "${MH[@]}" --algo grpo --steps 501 --ppo_epochs 2 --clip_eps 0.2 ;;
  # --- GRPO ablations on format_copy (later flags override earlier ones in argparse) ---
  abl_epochs1)   run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 1 --grad_accum_steps 6 --clip_eps 0.2 ;;
  abl_epochs4)   run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 4 --grad_accum_steps 6 --clip_eps 0.2 ;;
  abl_kl0)       run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 2 --grad_accum_steps 6 --clip_eps 0.2 --kl_coef 0.0 ;;
  abl_kl1)       run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 2 --grad_accum_steps 6 --clip_eps 0.2 --kl_coef 1.0 ;;
  abl_kl10)      run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 2 --grad_accum_steps 6 --clip_eps 0.2 --kl_coef 10.0 ;;
  abl_clip0.05)  run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 2 --grad_accum_steps 6 --clip_eps 0.05 ;;
  abl_clip0.5)   run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 2 --grad_accum_steps 6 --clip_eps 0.5 ;;
  abl_accum1)    run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 2 --grad_accum_steps 1 --clip_eps 0.2 ;;
  abl_mb48_accum1) run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 2 --grad_accum_steps 1 --clip_eps 0.2 --minibatch_size 48 ;;
  abl_mb16_accum3) run "${COMMON[@]}" "${FC[@]}" --algo grpo --ppo_epochs 2 --grad_accum_steps 3 --clip_eps 0.2 --minibatch_size 16 ;;
  *) echo "unknown run name: $NAME" >&2; exit 1 ;;
esac
