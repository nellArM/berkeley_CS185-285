"""Make the report figures for HW4 from runs/<name>/metrics.jsonl.

Usage (from the hw4 root):
    uv run --with matplotlib python scripts/plot_report.py [--runs runs] [--out report]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#7b3294", "#8c564b", "#e7298a"]
REWARD = "rollout/mean_total_reward_across_all_completions_in_batch_and_groups"
KL = "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches"
CLIP = "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches"
TRUNC = "rollout/fraction_of_completions_that_hit_max_new_tokens_limit"
EVAL_FC = "eval/format_copy_fraction_predicted_number_matches_target_integer_exactly"
EVAL_MATH_PREFIX = "eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer"


def load(runs, name):
    """Return {step: merged metrics dict} for one run."""
    out = {}
    with open(os.path.join(runs, name, "metrics.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            out.setdefault(row["step"], {}).update(row["metrics"])
    return out


def series(run, key, prefix=False):
    xs, ys = [], []
    for step in sorted(run):
        keys = [k for k in run[step] if k.startswith(key)] if prefix else ([key] if key in run[step] else [])
        if keys:
            xs.append(step)
            ys.append(run[step][keys[0]])
    return xs, ys


def smooth(ys, window):
    out = []
    for i in range(len(ys)):
        chunk = ys[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(out, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="report")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    L = lambda name: load(args.runs, name)

    # 1. format_copy: GRPO vs GR-REINFORCE
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for c, (name, label) in zip(COLORS, [("format_copy_grpo", "GRPO"), ("format_copy_reinforce", "GR-REINFORCE")]):
        run = L(name)
        axes[0].plot(*series(run, REWARD), color=c, label=label)
        axes[1].plot(*series(run, EVAL_FC), color=c, marker="o", ms=4, label=label)
    style(axes[0], "Training iteration", "Mean rollout reward", "(a) format_copy: training reward")
    style(axes[1], "Training iteration", "Eval exact match", "(b) format_copy: eval exact match")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    save(fig, args.out, "format_copy.png")

    # 2. math_hard: GRPO vs GR-REINFORCE
    runs = [("math_hard_grpo", "GRPO (501 iterations)"), ("math_hard_reinforce", "GR-REINFORCE (201 iterations)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for c, (name, label) in zip(COLORS, runs):
        run = L(name)
        xs, ys = series(run, REWARD)
        axes[0].plot(xs, ys, color=c, alpha=0.2, lw=0.8)
        axes[0].plot(xs, smooth(ys, 20), color=c, label=label)
        axes[1].plot(*series(run, EVAL_MATH_PREFIX, prefix=True), color=c, marker="o", ms=4, label=label)
    axes[0].axvline(200, color="gray", ls="--", lw=1)
    axes[1].axvline(200, color="gray", ls="--", lw=1)
    style(axes[0], "Training iteration", "Mean rollout reward", "(a) math_hard: training reward (20-iteration mean in bold)")
    style(axes[1], "Training iteration", "Eval exact match (boxed parser)", "(b) math_hard: held-out exact match")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    save(fig, args.out, "math_hard.png")

    # 3. math_hard diagnostics: KL and truncation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for c, (name, label) in zip(COLORS, runs):
        run = L(name)
        axes[0].plot(*series(run, KL), color=c, label=label)
        xs, ys = series(run, TRUNC)
        axes[1].plot(xs, smooth(ys, 20), color=c, label=label)
    axes[0].set_yscale("log")
    style(axes[0], "Training iteration", "Approximate KL to reference (log scale)", "(a) math_hard: sampled-token KL")
    style(axes[1], "Training iteration", "Fraction of completions", "(b) Completions cut off at 512 tokens (20-iteration mean)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    save(fig, args.out, "math_hard_diagnostics.png")

    # 4. GRPO ablations on format_copy: one row of panels per hyperparameter family
    families = [
        ("ppo_epochs", [("abl_epochs1", "1"), ("format_copy_grpo", "2 (default)"), ("abl_epochs4", "4")]),
        ("kl_coef", [("abl_kl0", "0"), ("format_copy_grpo", "0.05 (default)"), ("abl_kl1", "1"), ("abl_kl10", "10")]),
        ("clip_eps", [("abl_clip0.05", "0.05"), ("format_copy_grpo", "0.2 (default)"), ("abl_clip0.5", "0.5")]),
        ("minibatch / grad_accum", [("format_copy_grpo", "8 / 6 (default)"), ("abl_accum1", "8 / 1"), ("abl_mb16_accum3", "16 / 3"), ("abl_mb48_accum1", "48 / 1")]),
    ]
    fig, axes = plt.subplots(len(families), 3, figsize=(11, 2.9 * len(families)))
    for row, (family, variants) in zip(axes, families):
        for c, (name, label) in zip(COLORS, variants):
            run = L(name)
            row[0].plot(*series(run, REWARD), color=c, label=label)
            row[1].plot(*series(run, KL), color=c, label=label)
            row[2].plot(*series(run, CLIP), color=c, label=label)
        style(row[0], "Training iteration", "Mean rollout reward", f"{family}: reward")
        style(row[1], "Training iteration", "Approximate KL", f"{family}: KL to reference")
        style(row[2], "Training iteration", "Clip fraction", f"{family}: clip fraction")
        row[0].legend(frameon=False, fontsize=7, title=family, title_fontsize=7, loc="center right" if family == "kl_coef" else "lower right")
    save(fig, args.out, "ablations.png")


if __name__ == "__main__":
    main()
