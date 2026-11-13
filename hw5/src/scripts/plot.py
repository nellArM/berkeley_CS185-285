"""Make the report figures for HW5 from the eval.csv files.

Usage (from the hw5 root):
    uv run src/scripts/plot.py [--runs exp_all] [--exp exp] [--out report]

`--runs` is the directory holding the raw run folders, laid out as
<runs>/<q1|q2|q3>/sd<seed>_<agent>_<env>_a<alpha>/eval.csv
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import json

COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#7b3294", "#8c564b", "#e7298a"]
ENVS = {"cube-single": "cube-single-play-singletask-task1-v0", "antsoccer-arena": "antsoccer-arena-navigate-singletask-task1-v0"}
METHODS = {"q1": ("sacbc", "SAC+BC"), "q2": ("iql", "IQL"), "q3": ("fql", "FQL")}


def load(runs, q, env):
    """Return {alpha: dataframe} for one question and environment."""
    out = {}
    for d in glob.glob(os.path.join(runs, q, f"sd*_{METHODS[q][0]}_{ENVS[env]}_a*")):
        path = os.path.join(d, "eval.csv")
        if os.path.exists(path):
            alpha = float(re.search(r"_a([0-9.]+)$", d).group(1))
            out[alpha] = read_eval(path)
    return dict(sorted(out.items()))


def read_eval(path):
    rows = list(csv.DictReader(open(path)))
    return {k: [float(r[k]) for r in rows] for k in ("step", "eval/success_rate")}


def load_submitted(exp, q, env):
    """The run chosen for submission: <exp>/<q>/<env>/{eval.csv, flags.json}."""
    d = os.path.join(exp, q, env)
    alpha = json.load(open(os.path.join(d, "flags.json")))["agent_kwargs"]["alpha"]
    return alpha, read_eval(os.path.join(d, "eval.csv"))


def style(ax, title):
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="exp_all")
    p.add_argument("--exp", default="exp")
    p.add_argument("--out", default="report")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for q, (_, label) in METHODS.items():
        # Best run on each task
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, env in zip(axes, ENVS):
            a, df = load_submitted(args.exp, q, env)
            ax.plot(df["step"], df["eval/success_rate"], color=COLORS[0], marker="o", ms=3)
            style(ax, f"{label} on {env} (submitted run, alpha = {a:g})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{q}_best.png"), dpi=150)
        plt.close(fig)

        # Alpha sweep on both tasks
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, env in zip(axes, ENVS):
            for c, (a, df) in zip(COLORS, load(args.runs, q, env).items()):
                ax.plot(df["step"], df["eval/success_rate"], color=c, marker="o", ms=3, label=f"alpha = {a:g}")
            ax.legend(frameon=False, fontsize=8)
            style(ax, f"{label} on {env}: alpha sweep")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{q}_alpha_sweep.png"), dpi=150)
        plt.close(fig)
        print("wrote figures for", q)


if __name__ == "__main__":
    main()
