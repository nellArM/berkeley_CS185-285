"""Collect the numbers and figures of the report.

Usage: uv run python scripts/make_report.py   (from the repository root, after training/judging)

Reads runs/*/metrics.jsonl, runs/judge/*/*_summary.json and runs/judge/rm_scores*.json, writes
report/*.png and prints the result tables in Markdown.
"""
import glob
import json
import os
import statistics as st

import sys

sys.path.insert(0, os.getcwd())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "runs/wildchat_min4_judged_5k_"
OUT = "report"


def metrics(run):
    rows = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
    return [(r["step"], r["metrics"]) for r in rows]


def series(run, key):
    xs, ys = [], []
    for step, m in metrics(run):
        if key in m:
            xs.append(step)
            ys.append(m[key])
    return xs, ys


def judge_summaries():
    out = {}
    for f in sorted(glob.glob("runs/judge/*/*_summary.json")):
        d = json.load(open(f))
        if d.get("mode") != "policy":
            continue
        name = os.path.basename(f)[: -len("_summary.json")]
        rows = [json.loads(l) for l in open(f.replace("_summary.json", "_judgments.jsonl"))]
        # Recompute from the raw judgments so that all files use the same definition.
        from llm_rl_final_proj.local_judge import summarize

        out[name] = summarize(rows)
    return out


def rm_scores():
    out = {}
    for f in sorted(glob.glob("runs/judge/rm_scores*.json")):
        out.update({k: st.mean(v) for k, v in json.load(open(f))["scores"].items()})
    return out


def gen_stats():
    out = {}
    files = glob.glob("llm_rl_final_proj_public_submission/*/*.jsonl") + glob.glob("runs/gens/*.jsonl")
    files.append("public_eval/public_test_gen_base_responses_128.jsonl")
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        if "response_text" not in rows[0]:
            continue
        toks = [r.get("generated_num_tokens", 0) for r in rows]
        out[os.path.basename(f)[:-6]] = (st.mean(toks), sum(t >= 256 for t in toks) / len(toks))
    return out


def fig_offline(runs, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    keys = [
        ("eval/reference_corrected_pref_accuracy", "Held-out accuracy of the reference-corrected margin"),
        ("eval/pref_accuracy_mean_logp", "Held-out accuracy of the raw mean log-prob margin"),
        ("eval/generation_mean_num_tokens", "Mean generated tokens (32 test prompts)"),
    ]
    for ax, (key, ylabel) in zip(axes, keys):
        for label, run in runs:
            xs, ys = series(run, key)
            if key.endswith("pref_accuracy"):
                xs, ys = xs[1:], ys[1:]  # step 0: policy = reference, the margin is identically 0
            ax.plot(xs, ys, marker="o", ms=3, label=label)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_reward_model(path):
    run = R + "reward_model_v1"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    xs, ys = series(run, "eval/rm_pair_accuracy")
    axes[0].plot(xs, ys, marker="o", ms=3)
    axes[0].axhline(0.74, color="k", ls="--", lw=1, label="handout threshold 0.74")
    axes[0].set_ylabel("Pair accuracy on test_prefs")
    axes[0].legend(fontsize=8)
    for key, label in [("eval/rm_margin_mean", "margin mean"), ("eval/rm_margin_std", "margin std")]:
        xs, ys = series(run, key)
        axes[1].plot(xs, ys, marker="o", ms=3, label=label)
    axes[1].set_ylabel("score(chosen) - score(rejected)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_online(runs, path):
    keys = [
        ("rollout/reward_model_score_mean", "Reward used for training (rollout mean)"),
        ("train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches", "Sampled KL to the reference"),
        ("rollout/completion_mean_tokens", "Mean rollout completion tokens"),
        ("train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches", "Clipped fraction"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (key, ylabel) in zip(axes, keys):
        for label, run in runs:
            xs, ys = series(run, key)
            if len(ys) > 5 and "KL" not in ylabel:  # light smoothing: running mean over 5 iterations
                ys = [st.mean(ys[max(0, i - 4) : i + 1]) for i in range(len(ys))]
            ax.plot(xs, ys, label=label)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        if "KL" in ylabel:
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_scatter(names, judge, rms, gens, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, label in names:
        if name not in judge or name not in rms:
            continue
        w = judge[name]["win_rate_debiased_all_rows"]
        axes[0].scatter(rms[name], w)
        axes[0].annotate(label, (rms[name], w), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axes[1].scatter(gens[name][0], w)
        axes[1].annotate(label, (gens[name][0], w), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("Mean reward-model score of the 128 public responses")
    axes[1].set_xlabel("Mean generated tokens (128 public prompts)")
    for ax in axes:
        ax.set_ylabel("Proxy win rate vs. base (debiased)")
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    judge, rms, gens = judge_summaries(), rm_scores(), gen_stats()

    fig_offline(
        [("DPO (beta 0.1)", R + "dpo_beta01_v1"), ("IPO (beta 0.1)", R + "ipo_v1"), ("AOT (beta 0.2)", R + "aot_beta02_v1")],
        f"{OUT}/offline_part1.png", "Part 1 offline methods")
    p2 = [("DPO (first epoch)", R + "dpo_beta01_v1"), ("+ confidence weighting", "runs/p2_dpo_cw"),
          ("+ DPO-Positive penalty", "runs/p2_dpop"), ("+ all three", "runs/p2_dpo_all")]
    fig_offline([x for x in p2 if os.path.exists(x[1])], f"{OUT}/offline_part2.png", "Part 2 offline variants (one epoch)")
    fig_reward_model(f"{OUT}/reward_model.png")
    fig_online([("GRPO", R + "grpo_rm445_v1"), ("DrGRPO", R + "drgrpo_rm445_v1"), ("GSPO", R + "gspo_rm445_v1")], f"{OUT}/online_part1.png")
    on2 = [("GRPO, 75 iterations", "runs/p2on_base75"), ("+ rank advantages", "runs/p2on_rank"),
           ("+ pessimistic ensemble", "runs/p2on_pess"), ("+ rank + pessimistic ensemble", "runs/p2on_rank_pess")]
    on2 = [x for x in on2 if os.path.exists(os.path.join(x[1], "metrics.jsonl"))]
    if on2:
        fig_online(on2, f"{OUT}/online_part2.png")
    labels = [(n, n.replace("_step", "@").replace("public_test_gen_base_responses_128", "base")) for n in sorted(judge)]
    fig_scatter(labels, judge, rms, gens, f"{OUT}/rm_vs_judge.png")

    print("| Output | Win rate (debiased, all rows) | Win rate (pair-agree) | Pair-agree rows | RM score | Tokens | At 256-token limit |")
    print("|---|---|---|---|---|---|---|")
    for n in sorted(judge):
        j = judge[n]
        g = gens.get(n, (float("nan"), float("nan")))
        print(f"| {n} | {j['win_rate_debiased_all_rows']:.3f} ± {j['win_rate_debiased_all_rows_stderr']:.3f} | "
              f"{j['win_rate_pair_agree']:.3f} | {j['count_pair_agree_rows']} | {rms.get(n, float('nan')):.2f} | {g[0]:.0f} | {g[1]:.2f} |")
    print(f"| base | 0.5 | | | {rms.get('public_test_gen_base_responses_128', float('nan')):.2f} | "
          f"{gens['public_test_gen_base_responses_128'][0]:.0f} | {gens['public_test_gen_base_responses_128'][1]:.2f} |")


if __name__ == "__main__":
    main()
