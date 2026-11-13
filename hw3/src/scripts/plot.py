"""Make the report figures for HW3 from the log.csv files.

Usage (from the hw3 root):
    uv run src/scripts/plot.py [--exp_dirs exp exp_extra] [--out report]
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#7b3294", "#8c564b"]


def find_run(exp_dirs, prefix):
    """Return the log.csv dataframe of the run named `<prefix>_sd<seed>_<ts>`.

    The directories are searched in order (the submitted runs in `exp` first); within a
    directory the latest run wins.
    """
    for d in exp_dirs:
        matches = glob.glob(os.path.join(d, prefix + "_sd*"))
        matches = [m for m in matches if os.path.basename(m).split("_sd")[0] == prefix]
        if matches:
            return pd.read_csv(os.path.join(sorted(matches)[-1], "log.csv"))
    raise FileNotFoundError(prefix)


def find_seed(exp_dirs, prefix, seed):
    for d in exp_dirs:
        matches = glob.glob(os.path.join(d, f"{prefix}_sd{seed}_*"))
        if matches:
            return pd.read_csv(os.path.join(sorted(matches)[-1], "log.csv"))
    raise FileNotFoundError(f"{prefix} seed {seed}")


def series(df, key):
    s = df[["step", key]].dropna()
    return s["step"].to_numpy(), s[key].to_numpy()


def smooth(y, window):
    return pd.Series(y).rolling(window, min_periods=1).mean().to_numpy()


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
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
    p.add_argument("--exp_dirs", nargs="+", default=["exp", "exp_extra"])
    p.add_argument("--out", default="report")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    E = args.exp_dirs

    # 1. CartPole
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(*series(find_run(E, "CartPole-v1_dqn"), "Eval_AverageReturn"), color=COLORS[0])
    ax.axhline(500, color="gray", ls="--", lw=1)
    style(ax, "Environment steps", "Eval return", "DQN on CartPole-v1")
    save(fig, args.out, "cartpole.png")

    # 2. LunarLander
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(*series(find_run(E, "LunarLander-v2_dqn"), "Eval_AverageReturn"), color=COLORS[0])
    ax.axhline(200, color="gray", ls="--", lw=1)
    style(ax, "Environment steps", "Eval return", "Double DQN on LunarLander-v2")
    save(fig, args.out, "lunarlander.png")

    # 3. MsPacman: train vs eval return
    df = find_run(E, "MsPacman_dqn")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x, y = series(df, "Train_EpisodeReturn")
    ax.plot(x, y, color=COLORS[1], alpha=0.25, lw=0.8)
    ax.plot(x, smooth(y, 50), color=COLORS[1], label="Train return (per episode, 50-episode mean in bold)")
    ax.plot(*series(df, "Eval_AverageReturn"), color=COLORS[0], label="Eval return (mean of 10 episodes)")
    ax.legend(frameon=False, fontsize=8)
    style(ax, "Environment steps", "Return", "Double DQN on MsPacman")
    save(fig, args.out, "mspacman.png")

    # 4. LunarLander learning-rate sweep
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    sweep = [("dqn_lr0.0001", "lr = 1e-4"), ("dqn_lr0.0003", "lr = 3e-4"), ("dqn", "lr = 1e-3 (default)"), ("dqn_lr0.003", "lr = 3e-3")]
    for c, (name, label) in zip(COLORS, sweep):
        ax.plot(*series(find_run(E, "LunarLander-v2_" + name), "Eval_AverageReturn"), color=c, label=label)
    ax.axhline(200, color="gray", ls="--", lw=1)
    ax.legend(frameon=False, fontsize=8)
    style(ax, "Environment steps", "Eval return", "LunarLander-v2: learning-rate sweep")
    save(fig, args.out, "lunarlander_lr_sweep.png")

    # 5. HalfCheetah, fixed temperature
    fixed = find_run(E, "HalfCheetah-v4_sac")
    auto = find_run(E, "HalfCheetah-v4_sac_autotune")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(*series(fixed, "Eval_AverageReturn"), color=COLORS[0])
    ax.axhline(6000, color="gray", ls="--", lw=1)
    style(ax, "Environment steps", "Eval return", "SAC on HalfCheetah-v4 (fixed temperature 0.1)")
    save(fig, args.out, "halfcheetah.png")

    # 6. HalfCheetah fixed vs auto-tuned + temperature
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(*series(fixed, "Eval_AverageReturn"), color=COLORS[0], label="Fixed temperature 0.1")
    axes[0].plot(*series(auto, "Eval_AverageReturn"), color=COLORS[1], label="Auto-tuned temperature")
    axes[0].legend(frameon=False, fontsize=8)
    style(axes[0], "Environment steps", "Eval return", "(a) HalfCheetah-v4 eval return")
    axes[1].plot(*series(auto, "temperature"), color=COLORS[1], label="Auto-tuned")
    axes[1].axhline(0.1, color=COLORS[0], ls="--", lw=1, label="Fixed (0.1)")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[1], "Environment steps", "Temperature (log scale)", "(b) Temperature during training")
    save(fig, args.out, "halfcheetah_autotune.png")

    # 7. Hopper single-Q vs clipped double-Q
    single = find_run(E, "Hopper-v4_sac_singleq")
    clip = find_run(E, "Hopper-v4_sac_clipq")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title, ylabel in [
        (axes[0], "Eval_AverageReturn", "(a) Hopper-v4 eval return", "Eval return"),
        (axes[1], "q_values", "(b) Mean predicted Q-value", "Q-value"),
    ]:
        ax.plot(*series(single, key), color=COLORS[0], label="Single Q")
        ax.plot(*series(clip, key), color=COLORS[1], label="Clipped double Q")
        ax.legend(frameon=False, fontsize=8)
        style(ax, "Environment steps", ylabel, title)
    save(fig, args.out, "hopper.png")

    # 8. Hopper: all three seeds of both variants (seeds 1 and 2 are in exp_extra)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title, ylabel in [
        (axes[0], "Eval_AverageReturn", "(a) Hopper-v4 eval return, seeds 1-3", "Eval return"),
        (axes[1], "q_values", "(b) Mean predicted Q-value, seeds 1-3", "Q-value"),
    ]:
        for seed, ls in zip((1, 2, 3), (":", "--", "-")):
            ax.plot(*series(find_seed(E, "Hopper-v4_sac_singleq", seed), key), color=COLORS[0], ls=ls, label=f"Single Q, seed {seed}")
            ax.plot(*series(find_seed(E, "Hopper-v4_sac_clipq", seed), key), color=COLORS[1], ls=ls, label=f"Clipped double Q, seed {seed}")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        style(ax, "Environment steps", ylabel, title)
    axes[0].axhline(1500, color="gray", ls="--", lw=1)
    save(fig, args.out, "hopper_seeds.png")


if __name__ == "__main__":
    main()
