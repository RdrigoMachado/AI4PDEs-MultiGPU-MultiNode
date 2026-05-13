#!/usr/bin/env python3
"""Generate publication-grade figures from Exp 1 results.

Reads exp1_summary.csv (produced by exp1_summary.py) and emits PDF figures
in <output_dir>. Designed for a single-column IEEE-style paper: serif font,
sober palette, vector output.

Figures produced:
    fig_exec_vs_nodes_<topology>.pdf       T_exec vs nodes per io_mode
    fig_overhead_vs_nodes_<topology>.pdf   I/O overhead (% over none) vs nodes
    fig_save_flush_breakdown.pdf           Save vs Flush time, async vs naive
    fig_weak_efficiency_<topology>.pdf     T(1) / T(N) per io_mode

Usage:
    python tools/exp1_plots.py path/to/exp1_summary.csv [--outdir figs]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Okabe-Ito colorblind-safe palette, one color per io_mode.
COLOR = {"none": "#000000", "naive": "#D55E00", "async": "#0072B2"}
MARKER = {"none": "o", "naive": "s", "async": "^"}
LABEL = {"none": "no I/O", "naive": "naive (sync)", "async": "async (per-rank pool)"}


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "lines.markersize": 5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_summary(path):
    """Load exp1_summary.csv into a list of dicts (numeric fields parsed)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["nodes"] = int(row["nodes"])
                row["n_runs"] = int(row["n_runs"])
                for k in ("exec_mean", "exec_std", "save_mean", "save_std",
                          "flush_mean", "flush_std"):
                    row[k] = float(row[k])
                row["overhead_pct"] = (float(row["overhead_pct"])
                                       if row["overhead_pct"] else None)
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def by_topology(rows, topology):
    return [r for r in rows if r["topology"] == topology]


def series(rows, io_mode, field):
    """Return (nodes, values) sorted by nodes for a given io_mode."""
    pts = sorted([(r["nodes"], r[field]) for r in rows if r["io_mode"] == io_mode])
    if not pts:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def fig_exec_vs_nodes(rows, topology, outdir):
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for io in ("none", "naive", "async"):
        x, y = series(rows, io, "exec_mean")
        _, ystd = series(rows, io, "exec_std")
        if len(x) == 0:
            continue
        ax.errorbar(x, y, yerr=ystd, color=COLOR[io], marker=MARKER[io],
                    label=LABEL[io], capsize=2, linewidth=1.3)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("nodes")
    ax.set_ylabel("execution time (s)")
    ax.set_xticks([1, 2, 4, 8, 16, 20])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "20"])
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    out = os.path.join(outdir, f"fig_exec_vs_nodes_{topology}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_overhead_vs_nodes(rows, topology, outdir):
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for io in ("naive", "async"):
        x, y = series(rows, io, "overhead_pct")
        if len(x) == 0:
            continue
        ax.plot(x, y, color=COLOR[io], marker=MARKER[io],
                label=LABEL[io], linewidth=1.3)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("nodes")
    ax.set_ylabel("I/O overhead (\\%)" if matplotlib.rcParams["text.usetex"]
                  else "I/O overhead (%)")
    ax.set_xticks([1, 2, 4, 8, 16, 20])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "20"])
    ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    out = os.path.join(outdir, f"fig_overhead_vs_nodes_{topology}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_save_flush_breakdown(rows, topology, outdir):
    """Stacked bars: Save_time + Flush_time for naive vs async at each node count."""
    naive_rows = sorted([r for r in rows
                         if r["io_mode"] == "naive" and r["topology"] == topology],
                        key=lambda r: r["nodes"])
    async_rows = sorted([r for r in rows
                         if r["io_mode"] == "async" and r["topology"] == topology],
                        key=lambda r: r["nodes"])
    nodes = sorted({r["nodes"] for r in naive_rows} | {r["nodes"] for r in async_rows})
    if not nodes:
        return None

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(nodes))
    width = 0.38

    def get(rows_list, n, field):
        for r in rows_list:
            if r["nodes"] == n:
                return r[field]
        return 0.0

    naive_save = [get(naive_rows, n, "save_mean") for n in nodes]
    naive_flush = [get(naive_rows, n, "flush_mean") for n in nodes]
    async_save = [get(async_rows, n, "save_mean") for n in nodes]
    async_flush = [get(async_rows, n, "flush_mean") for n in nodes]

    ax.bar(x - width/2, naive_save, width, color=COLOR["naive"], label="naive — Save",
           edgecolor="black", linewidth=0.4)
    ax.bar(x - width/2, naive_flush, width, bottom=naive_save,
           color=COLOR["naive"], alpha=0.4, hatch="//",
           label="naive — Flush", edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, async_save, width, color=COLOR["async"], label="async — Save",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, async_flush, width, bottom=async_save,
           color=COLOR["async"], alpha=0.4, hatch="//",
           label="async — Flush", edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nodes])
    ax.set_xlabel("nodes")
    ax.set_ylabel("time on critical path (s)")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    out = os.path.join(outdir, f"fig_save_flush_breakdown_{topology}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_weak_efficiency(rows, topology, outdir):
    """Weak-scaling efficiency = T(1) / T(N), one curve per io_mode."""
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for io in ("none", "naive", "async"):
        x, y = series(rows, io, "exec_mean")
        if len(x) == 0:
            continue
        t1 = y[0] if x[0] == 1 else None
        if t1 is None:
            continue
        eff = t1 / y
        ax.plot(x, eff, color=COLOR[io], marker=MARKER[io],
                label=LABEL[io], linewidth=1.3)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("nodes")
    ax.set_ylabel("weak-scaling efficiency")
    ax.set_xticks([1, 2, 4, 8, 16, 20])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "20"])
    ax.axhline(1.0, color="grey", linewidth=0.4, linestyle=":")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    out = os.path.join(outdir, f"fig_weak_efficiency_{topology}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary_csv", help="Path to exp1_summary.csv")
    ap.add_argument("--outdir", default="figs",
                    help="Directory for the figures (default: figs/)")
    args = ap.parse_args()

    if not os.path.isfile(args.summary_csv):
        print(f"ERRO: {args.summary_csv} não encontrado.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    setup_style()
    rows = load_summary(args.summary_csv)
    if not rows:
        print("Nenhuma linha no summary.", file=sys.stderr)
        sys.exit(1)

    topologies = sorted({r["topology"] for r in rows})
    written = []
    for topo in topologies:
        sub = by_topology(rows, topo)
        written.append(fig_exec_vs_nodes(sub, topo, args.outdir))
        written.append(fig_overhead_vs_nodes(sub, topo, args.outdir))
        written.append(fig_weak_efficiency(sub, topo, args.outdir))
        out = fig_save_flush_breakdown(sub, topo, args.outdir)
        if out:
            written.append(out)

    for w in written:
        if w:
            print(f"escrito: {w}")


if __name__ == "__main__":
    main()
