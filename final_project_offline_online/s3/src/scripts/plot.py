"""Make the report figures from the raw runs.

Usage: uv run src/scripts/plot.py [--runs exp_all] [--out report]

Every run directory holds eval.csv (eval/success_rate, step), train.csv (logged metrics, `phase`
0 = offline / 1 = online, step) and flags.json. Runs are looked up by run group, seed and a few
substrings of the directory name; see RUNS at the bottom of this file.
"""
import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OFFLINE_STEPS = 500_000
ENVS = {
    "single": "cube-single-play",
    "double": "cube-double-play",
    "soccer": "antsoccer-arena-navigate",
}
TITLES = {"single": "cube-single", "double": "cube-double", "soccer": "antsoccer-arena"}


def find(root, group, env, *keys, seed=0, exclude=()):
    """Return the unique run dir in `root/group` for `env` and `seed` whose name contains all `keys`."""
    hits = []
    for d in sorted(glob.glob(os.path.join(root, group, f"sd{seed}_*"))):
        name = os.path.basename(d).replace("_online_offline", "") + "_"
        if ENVS[env] not in name:
            continue
        if all(f"_{k}_" in name for k in keys) and not any(f"_{k}" in name for k in exclude):
            if os.path.exists(os.path.join(d, "eval.csv")):
                hits.append(d)
    if len(hits) != 1:
        raise RuntimeError(f"{group}/{env}/sd{seed}/{keys}: expected 1 run, found {hits}")
    return hits[0]


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        cols[k] = np.array([float(v) if v not in ("", None) else np.nan for v in vals])
    return cols


def success(run):
    c = read_csv(os.path.join(run, "eval.csv"))
    return c["step"], c["eval/success_rate"]


def boundary(ax):
    ax.axvline(OFFLINE_STEPS, color="k", ls="--", lw=1)
    ax.axvspan(OFFLINE_STEPS, OFFLINE_STEPS + 100_000, color="0.92", zorder=0)


def fmt(ax, title, xlim=None, ylabel="Evaluation success rate"):
    ax.set_title(title)
    ax.set_xlabel("Gradient / environment step")
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.grid(alpha=0.3)


def plot_success_panels(spec, envs, path, xlim=(0, 600_000), ncol_legend=1):
    """spec: list of (label, {env: [run dirs (one per seed)]}). Mean over seeds, band = min/max."""
    fig, axes = plt.subplots(1, len(envs), figsize=(6.2 * len(envs), 4.2), squeeze=False)
    for ax, env in zip(axes[0], envs):
        for i, (label, runs) in enumerate(spec):
            if not runs.get(env):
                continue
            curves = [success(r) for r in runs[env]]
            steps = curves[0][0]
            n = min(len(c[0]) for c in curves)
            steps = steps[:n]
            ys = np.stack([c[1][:n] for c in curves])
            color = f"C{i}"
            ax.plot(steps, ys.mean(0), marker="o", ms=3, color=color, label=label)
            if len(curves) > 1:
                ax.fill_between(steps, ys.min(0), ys.max(0), color=color, alpha=0.15)
        boundary(ax)
        ax.set_ylim(-0.03, 1.03)
        fmt(ax, TITLES[env], xlim)
    axes[0][-1].legend(fontsize=8, ncol=ncol_legend, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def plot_metrics(spec, metrics, path, xlim=(400_000, 600_000)):
    """spec: list of (label, run, {metric title: column}); one row per run, one column per metric."""
    fig, axes = plt.subplots(len(spec), len(metrics), figsize=(4.3 * len(metrics), 3.2 * len(spec)), squeeze=False)
    for row, (label, run, cols) in zip(axes, spec):
        c = read_csv(os.path.join(run, "train.csv"))
        for ax, title in zip(row, metrics):
            col = cols[title]
            for j, name in enumerate(col if isinstance(col, (list, tuple)) else [col]):
                ax.plot(c["step"], c[name], color=f"C{j}", label=name)
            boundary(ax)
            fmt(ax, f"{label}: {title}", xlim, ylabel=title)
            m = (c["step"] >= xlim[0]) & (c["step"] <= xlim[1])
            if isinstance(col, (list, tuple)):
                ax.legend(fontsize=7)
            elif "loss" in title.lower() and np.nanmin(c[col][m]) > 0:
                ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="exp_all")
    p.add_argument("--out", default="report")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = a.runs

    def f(*args, **kw):
        return find(R, *args, **kw)

    # ---- Part I: naive offline-to-online on cube-single -------------------------------------
    sac = f("s1_sacbc", "single", "a300.0")
    fql = f("s1_fql", "single", "a300.0")
    plot_success_panels(
        [("SAC+BC (alpha=300)", {"single": [sac]}), ("FQL (alpha=300)", {"single": [fql]}),
         ("SAC+BC (alpha=100)", {"single": [f("s1_sacbc", "single", "a100.0")]}),
         ("FQL (alpha=100)", {"single": [f("s1_fql", "single", "a100.0")]})],
        ["single"], os.path.join(a.out, "part1_success.png"))
    plot_metrics(
        [("SAC+BC", sac, {"critic loss": "critic/q_loss", "Q-values": ["critic/q_mean", "critic/q_min", "critic/q_max"],
                          "actor loss terms": ["actor/q_loss", "actor/bc_loss"], "action MSE to data": "actor/mse"}),
         ("FQL", fql, {"critic loss": "critic/q_loss", "Q-values": ["critic/q_mean", "critic/q_min", "critic/q_max"],
                       "actor loss terms": ["onestep_actor/q_loss", "onestep_actor/distill_loss", "bc_actor/loss"],
                       "action MSE to data": "onestep_actor/mse"})],
        ["critic loss", "Q-values", "actor loss terms", "action MSE to data"],
        os.path.join(a.out, "part1_metrics.png"))

    # Naive FQL on the two harder tasks (collapse), with its Q-values.
    nd, ns = f("base_fql", "double", "a300.0"), f("base_fql", "soccer", "a10.0")
    plot_metrics(
        [("FQL cube-double", nd, {"critic loss": "critic/q_loss", "Q-values": ["critic/q_mean", "critic/q_min", "critic/q_max"],
                                  "action MSE to data": "onestep_actor/mse"}),
         ("FQL antsoccer", ns, {"critic loss": "critic/q_loss", "Q-values": ["critic/q_mean", "critic/q_min", "critic/q_max"],
                                "action MSE to data": "onestep_actor/mse"})],
        ["critic loss", "Q-values", "action MSE to data"], os.path.join(a.out, "naive_hard_metrics.png"))

    # ---- Part II: offline data retention and WSRL ---------------------------------------------
    naive = {"double": [nd], "soccer": [ns]}
    for tag, group, name, vals in [("od", "s2_offline", "offline transitions", (10000, 100000, 1000000)),
                                   ("ws", "s2_wsrl", "warm-up steps", (1000, 5000, 20000))]:
        spec = [(f"0 {name} (naive FQL)", naive)]
        for v in vals:
            spec.append((f"{v:,} {name}", {e: [f(group, e, f"{tag}{v}")] for e in ("double", "soccer")}))
        plot_success_panels(spec, ["double", "soccer"], os.path.join(a.out, f"part2_{group[3:]}.png"), xlim=(400_000, 600_000))

    # ---- Part II: IFQL / DSRL / QSM on the three tasks (seed 0) ----------------------------------
    spec = [
        ("IFQL (tau=0.9, N=32)", {e: [f("s2_ifql", e, "e0.9")] for e in ENVS}),
        ("DSRL (noise scale 1)", {e: [f("s2_dsrl", e, "n1.0")] for e in ENVS}),
        ("QSM", {e: [f(QSM_GROUP.get(e, "s2_qsm"), e, *QSM[e])] for e in ENVS}),
    ]
    plot_success_panels(spec, list(ENVS), os.path.join(a.out, "part2_methods.png"))

    # ---- Part III: all seven methods, two seeds ---------------------------------------------------
    def two(fn):
        out = {}
        for e in ("double", "soccer"):
            out[e] = []
            for seed in (0, 1):
                try:
                    run = fn(e, seed)
                    success(run)[1][-1]
                    if success(run)[0][-1] < OFFLINE_STEPS + 100_000:
                        raise RuntimeError(f"{run}: not finished")
                    out[e].append(run)
                except RuntimeError as err:
                    print("WARNING (run skipped):", err)
        return out

    A = {"double": "a300.0", "soccer": "a10.0"}
    spec = [
        ("FQL (naive)", two(lambda e, s: f("base_fql" if s == 0 else "s3_fql", e, A[e], seed=s))),
        ("FQL + offline data", two(lambda e, s: f("s2_offline" if s == 0 else "s3_offline", e, f"od{BEST_OD[e]}", seed=s))),
        ("FQL + WSRL", two(lambda e, s: f("s2_wsrl" if s == 0 else "s3_wsrl", e, f"ws{BEST_WS[e]}", seed=s))),
        ("IFQL", two(lambda e, s: f("s2_ifql" if s == 0 else "s3_ifql", e, "e0.9", seed=s))),
        ("DSRL", two(lambda e, s: f("s2_dsrl" if s == 0 else "s3_dsrl", e, "n1.0", seed=s))),
        ("QSM", two(lambda e, s: f(QSM_GROUP.get(e, "s2_qsm") if s == 0 else "s3_qsm", e, *QSM[e], seed=s))),
        ("Ours", two(lambda e, s: f("s3_yours", e, seed=s))),
    ]
    plot_success_panels(spec, ["double", "soccer"], os.path.join(a.out, "part3_all.png"), ncol_legend=2)
    plot_success_panels(spec, ["double", "soccer"], os.path.join(a.out, "part3_online.png"), xlim=(450_000, 600_000), ncol_legend=2)

    # Table of final numbers (per seed; "online mean" averages the five online evaluations).
    print("\nmethod | env | offline@500k (seeds) | final@600k (seeds) | mean final | online mean")
    for label, runs in spec:
        for e in ("double", "soccer"):
            off, fin, onl = [], [], []
            for r in runs[e]:
                s, y = success(r)
                off.append(y[s == OFFLINE_STEPS][0])
                fin.append(y[-1])
                onl.append(y[s > OFFLINE_STEPS].mean())
            fl = lambda v: "/".join(f"{x:.2f}" for x in v) if v else "n/a"
            mean = f"{np.mean(fin):.2f}" if fin else "n/a"
            omean = f"{np.mean(onl):.2f}" if onl else "n/a"
            print(f"{label:20s} {TITLES[e]:16s} {fl(off):10s} {fl(fin):10s} {mean:5s} {omean}")


# Settings selected from the seed-0 sweeps (see report).
QSM = {"single": ("a100.0", "i10.0"), "double": ("a100.0", "i10.0"), "soccer": ("a30.0", "i30.0", "mean_target1")}
QSM_GROUP = {"soccer": "s2_qsm_mean"}  # antsoccer: QSM with the mean critic target (see report)
BEST_OD = {"double": 100000, "soccer": 1000000}
BEST_WS = {"double": 20000, "soccer": 20000}

if __name__ == "__main__":
    main()
