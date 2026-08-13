#!/usr/bin/env python3
"""Generate figures for the paper.

Two figures:
  1. c_local_vs_nodes.png   — C_local vs node count, one curve per hardware config.
  2. c_quality_vs_nodes.png — C_quality vs node count, same configs.

At each node count N, the simulator picks the best (mode, model) for the chosen
metric, at 100 Mbps / 100 ms across all channels, 740-day training window,
subject to the same 1x-100x overtraining constraint as the paper tables.

Run from repo root:  python generate_paper_figures.py
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_paper_tables import (
    MODEL_SIZES, OT_MIN, OT_MAX, TIME_S, try_all_modes, _cq,
)
from evasion_calculator import CONFIGS, CONFIGS_FP8

BW = 100e6
LAT = 0.1

# Node counts to sweep (log-spaced)
NODE_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Curated config subset (name → (cfg_dict, linestyle, color, marker))
PLOT_CFGS = [
    ("16x H100 FP8",     CONFIGS_FP8["16x H100 FP8"],   "-",  "#1f77b4", "o"),
    ("16x GH200 FP8",    CONFIGS_FP8["16x GH200 FP8"],  "-",  "#ff7f0e", "s"),
    ("17x TPU v6e FP8",  CONFIGS_FP8["17x TPU v6e FP8"],"-",  "#2ca02c", "^"),
    ("50x A100 80GB",    CONFIGS["50x A100 80GB"],      "--", "#d62728", "D"),
    ("26x Ascend 910C",  CONFIGS["26x Ascend 910C"],    "--", "#9467bd", "v"),
    ("80x TPU v5e",      CONFIGS["80x TPU v5e"],        "--", "#8c564b", "P"),
]

# Paper-table C_local targets — horizontal reference lines
TARGETS = [1e24, 3.3e24, 1e25, 2.1e25, 3.8e25, 6.6e25, 1e26]


def best_at_nodes_for_metric(cfg, n_nodes, metric):
    """Return the (mode, r) at (cfg, n_nodes) maximizing 'c_local' or 'c_quality'
    under the OT constraint."""
    assert metric in ("c_local", "c_quality")
    best_mode, best_r, best_v = None, None, -1.0
    for pb in MODEL_SIZES:
        for mode, r in try_all_modes(cfg, n_nodes, pb, BW, LAT):
            ot = r.get("overtraining_ratio", 0)
            if ot < OT_MIN or ot > OT_MAX:
                continue
            v = r["c_local"] if metric == "c_local" else _cq(r)
            if v > best_v:
                best_v, best_mode, best_r = v, mode, r
    return best_mode, best_r, best_v


def sweep(cfg, metric):
    xs, ys = [], []
    for n in NODE_COUNTS:
        _, r, v = best_at_nodes_for_metric(cfg, n, metric)
        if r is None:
            continue
        xs.append(n)
        ys.append(v)
    return xs, ys


def fmt_target(t):
    exp = int(math.floor(math.log10(t)))
    coeff = t / 10**exp
    return f"$10^{{{exp}}}$" if abs(coeff - 1.0) < 0.01 else f"${coeff:.1f}\\times10^{{{exp}}}$"


def plot_metric(metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, cfg, ls, color, marker in PLOT_CFGS:
        xs, ys = sweep(cfg, metric)
        ax.plot(xs, ys, ls, color=color, marker=marker, markersize=5,
                linewidth=1.8, label=name, alpha=0.9)

    if metric == "c_local":
        for t in TARGETS:
            ax.axhline(t, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
            ax.text(NODE_COUNTS[-1]*1.05, t, fmt_target(t),
                    fontsize=7, color="gray", va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax.set_title(
        f"{ylabel} vs nodes (740 days, 100 Mbps, 100 ms; best over modes + model sizes)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    print(f"wrote {filename}")


if __name__ == "__main__":
    plot_metric("c_local",   "$C_{\\mathrm{local}}$ (FLOP)",   "c_local_vs_nodes.png")
    plot_metric("c_quality", "$C_{\\mathrm{quality}}$ (FLOP)", "c_quality_vs_nodes.png")
