"""Make the report figures from the log.csv files of the experiment runs.

Usage (from the hw2 directory):
    uv run src/scripts/plot.py [--exp_dirs exp exp_extra] [--out report]

Runs are located by their directory prefix `<env>_<exp_name>_sd` in the first of --exp_dirs that contains one; if
several runs there share a prefix, the newest is used.
The x-axis is always the number of environment steps (Train_EnvstepsSoFar).
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical palette, assigned in fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.size": 10,
    }
)


def load(exp_dirs, env, name):
    """Return the log of the newest run `<env>_<name>_sd*` as a dict of float columns."""
    matches = []
    for d in exp_dirs:
        # earlier directories take priority (exp holds the submitted runs, exp_extra the additional ones)
        matches = glob.glob(os.path.join(d, f"{env}_{name}_sd*"))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(f"no run found for {env}_{name}")
    # directory names end with a sortable timestamp
    run = max(matches, key=lambda p: os.path.basename(p).split("_sd")[-1].split("_", 1)[-1])
    with open(os.path.join(run, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows if r[k] != ""] for k in rows[0]}


def plot(exp_dirs, out, fname, title, env, series, y="Eval_AverageReturn", ylabel="Eval average return",
         hline=None, xmax=None, logy=False):
    """series: list of (exp_name, legend label, color index)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    for name, label, ci in series:
        log = load(exp_dirs, env, name)
        ax.plot(log["Train_EnvstepsSoFar"], log[y], color=COLORS[ci], label=label)
    if hline is not None:
        ax.axhline(hline, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"target {hline:g}", xy=(1, hline), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Environment steps (Train_EnvstepsSoFar)")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="plain")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}K" if v else "0"))
    if len(series) > 1:
        # below the plot, so the legend never covers a curve
        fig.legend(loc="lower center", ncol=min(len(series), 5 if len(series) == 5 else 2), labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06 * (1 + (len(series) == 4)) * (len(series) > 1), 1, 1))
    path = os.path.join(out, fname)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", default=["exp", "exp_extra"])
    parser.add_argument("--out", default="report")
    parser.add_argument("--pendulum_best", default="pendulum_tC", help="exp_name of the tuned pendulum run")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d, o = args.exp_dirs, args.out

    # the same configuration keeps the same color in both CartPole plots
    for suffix, bs in (("", 1000), ("_lb", 4000)):
        plot(d, o, f"cartpole{suffix}.png", f"CartPole-v0, batch size {bs}", "CartPole-v0",
             [(f"cartpole{suffix}", "vanilla", 0),
              (f"cartpole{suffix}_rtg", "reward-to-go", 1),
              (f"cartpole{suffix}_na", "normalized advantages", 2),
              (f"cartpole{suffix}_rtg_na", "reward-to-go + normalized advantages", 3)],
             hline=200)

    cheetah = [("cheetah", "no baseline", 0),
               ("cheetah_baseline", "baseline (-bgs 5)", 1),
               ("cheetah_baseline_bgs1", "baseline (-bgs 1)", 2)]
    plot(d, o, "cheetah_return.png", "HalfCheetah-v4: eval return", "HalfCheetah-v4", cheetah, hline=300)
    plot(d, o, "cheetah_baseline_loss.png", "HalfCheetah-v4: baseline loss", "HalfCheetah-v4", cheetah[1:],
         y="Baseline Loss", ylabel="Baseline loss (MSE)")

    plot(d, o, "lunar_lander.png", "LunarLander-v2: GAE λ sweep", "LunarLander-v2",
         [(f"lunar_lander_lambda{lam}", f"λ = {lam}", i) for i, lam in enumerate(["0", "0.95", "0.98", "0.99", "1"])],
         hline=150)

    plot(d, o, "pendulum.png", "InvertedPendulum-v4: default vs. tuned", "InvertedPendulum-v4",
         [("pendulum", "default (-b 5000)", 0), (args.pendulum_best, "tuned", 1)], hline=1000)


if __name__ == "__main__":
    main()
