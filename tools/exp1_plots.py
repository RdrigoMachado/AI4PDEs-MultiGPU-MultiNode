#!/usr/bin/env python3
"""Publication figures for Exp 1 — Weak Scaling with I/O.

Reads exp1_results.csv and emits PDF figures per topology, styled for a
single-column IEEE paper (serif, sober palette, vector output). The
'none' baseline is drawn as a dotted black line on most figures so the
reader sees instantly how each I/O mode tracks (or departs from) the
ideal weak-scaling reference.

Usage:
    python tools/exp1_plots.py path/to/exp1_results.csv [--outdir figs]
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_summary import load_runs, aggregate, derived  # noqa: E402


# Okabe-Ito colorblind-safe.
COLOR = {"none": "#000000", "naive": "#D55E00", "async": "#0072B2"}
MARKER = {"none": "o", "naive": "s", "async": "^"}
LABEL = {"none": "no I/O (baseline)", "naive": "naive (sync)", "async": "async"}


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sort_by_nodes(items, key="nodes"):
    return sorted(items, key=lambda r: r[key])


def series(rows, topology, io_mode, field):
    sub = sort_by_nodes([r for r in rows
                         if r["topology"] == topology and r["io_mode"] == io_mode])
    return (np.array([r["nodes"] for r in sub]),
            np.array([r[field] for r in sub]))


def io_modes_present(rows, topology):
    return [io for io in ("none", "naive", "async")
            if any(r for r in rows if r["topology"] == topology and r["io_mode"] == io)]


def setup_node_axis(ax, all_nodes):
    nodes = sorted(set(all_nodes))
    ax.set_xscale("log", base=2)
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(n) for n in nodes])
    ax.set_xlim(min(nodes) * 0.85, max(nodes) * 1.18)
    ax.set_xlabel("nodes")


def fig_exec_vs_nodes(rows, topology, outdir):
    """Headline figure: execution time vs nodes; baseline as dotted line."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    all_nodes = []

    # Baseline first (so it sits underneath the other curves visually).
    for io in ("none", "naive", "async"):
        if io not in io_modes_present(rows, topology):
            continue
        x, y = series(rows, topology, io, "exec_median")
        _, q1 = series(rows, topology, io, "exec_q1")
        _, q3 = series(rows, topology, io, "exec_q3")
        all_nodes.extend(x)
        style = {"color": COLOR[io], "marker": MARKER[io], "label": LABEL[io]}
        if io == "none":
            style.update({"linestyle": ":", "linewidth": 1.8})
        ax.plot(x, y, **style)
        ax.fill_between(x, q1, q3, color=COLOR[io], alpha=0.12, linewidth=0)

    setup_node_axis(ax, all_nodes)
    ax.set_ylabel("execution time (s)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    out = os.path.join(outdir, f"fig_exec_vs_nodes_{topology}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_overhead_vs_nodes(rows, topology, outdir):
    """I/O overhead (%) over baseline — naive and async only."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    all_nodes = []
    for io in ("naive", "async"):
        if io not in io_modes_present(rows, topology):
            continue
        x, y = series(rows, topology, io, "overhead_pct")
        all_nodes.extend(x)
        ax.plot(x, y, color=COLOR[io], marker=MARKER[io], label=LABEL[io])

    setup_node_axis(ax, all_nodes)
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2,
               label="baseline (no I/O)")
    ax.set_ylabel("I/O overhead vs baseline (%)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    out = os.path.join(outdir, f"fig_overhead_vs_nodes_{topology}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_weak_efficiency(rows, topology, outdir):
    """Weak-scaling efficiency = T(1, none) / T(N, io_mode)."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    all_nodes = []
    for io in ("none", "naive", "async"):
        if io not in io_modes_present(rows, topology):
            continue
        x, y = series(rows, topology, io, "weak_efficiency")
        all_nodes.extend(x)
        style = {"color": COLOR[io], "marker": MARKER[io], "label": LABEL[io]}
        if io == "none":
            style.update({"linestyle": ":", "linewidth": 1.8})
        ax.plot(x, y, **style)

    setup_node_axis(ax, all_nodes)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    ax.set_ylabel("weak-scaling efficiency  T(1) / T(N)")
    ax.set_ylim(0.5, 1.1)
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    out = os.path.join(outdir, f"fig_weak_efficiency_{topology}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_save_flush_breakdown(rows, topology, outdir):
    """Stacked bars: Save (solid) + Flush (hatched) for naive vs async."""
    naive = sort_by_nodes([r for r in rows
                           if r["topology"] == topology and r["io_mode"] == "naive"])
    asyn = sort_by_nodes([r for r in rows
                          if r["topology"] == topology and r["io_mode"] == "async"])
    nodes = sorted({r["nodes"] for r in naive} | {r["nodes"] for r in asyn})
    if not nodes:
        return None

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(nodes))
    width = 0.38

    def lookup(rs, n, field):
        for r in rs:
            if r["nodes"] == n:
                return r[field]
        return 0.0

    n_save = [lookup(naive, n, "save_median") for n in nodes]
    n_flush = [lookup(naive, n, "flush_median") for n in nodes]
    a_save = [lookup(asyn, n, "save_median") for n in nodes]
    a_flush = [lookup(asyn, n, "flush_median") for n in nodes]

    ax.bar(x - width/2, n_save, width, color=COLOR["naive"],
           label="naive — Save", edgecolor="black", linewidth=0.5)
    ax.bar(x - width/2, n_flush, width, bottom=n_save, color="none",
           edgecolor=COLOR["naive"], hatch="//", linewidth=0.5,
           label="naive — Flush")
    ax.bar(x + width/2, a_save, width, color=COLOR["async"],
           label="async — Save", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, a_flush, width, bottom=a_save, color="none",
           edgecolor=COLOR["async"], hatch="//", linewidth=0.5,
           label="async — Flush")

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nodes])
    ax.set_xlabel("nodes")
    ax.set_ylabel("time on critical path (s)")
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0,
              handletextpad=0.4)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    out = os.path.join(outdir, f"fig_save_flush_breakdown_{topology}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_boxplot_distribution(runs, topology, outdir):
    """Per-run distribution boxplot — exposes tails (Lustre variability)."""
    groups = defaultdict(list)
    for r in runs:
        if r["topology"] != topology:
            continue
        groups[(r["nodes"], r["io_mode"])].append(r["exec_s"])
    if not groups:
        return None

    nodes = sorted({n for (n, _) in groups})
    ios = ("none", "naive", "async")
    width = 0.24
    positions = {io: i for i, io in enumerate(ios)}

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for io in ios:
        data = []
        x_pos = []
        for i, n in enumerate(nodes):
            vals = groups.get((n, io), [])
            if vals:
                data.append(vals)
                x_pos.append(i + (positions[io] - 1) * width)
        if not data:
            continue
        bp = ax.boxplot(data, positions=x_pos, widths=width,
                        patch_artist=True, manage_ticks=False,
                        showfliers=True, whis=1.5)
        for box in bp["boxes"]:
            box.set(facecolor=COLOR[io], edgecolor="black",
                    linewidth=0.6, alpha=0.55)
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.0)
        for whisker in bp["whiskers"]:
            whisker.set(color="black", linewidth=0.6)
        for cap in bp["caps"]:
            cap.set(color="black", linewidth=0.6)
        for flier in bp["fliers"]:
            flier.set(marker="o", markerfacecolor=COLOR[io],
                      markeredgecolor="black", markersize=3.5,
                      linestyle="none", alpha=0.8)

    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([str(n) for n in nodes])
    ax.set_xlabel("nodes")
    ax.set_ylabel("execution time (s)")
    # Custom legend (boxes are bare colors).
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR[io], edgecolor="black", alpha=0.55,
                     label=LABEL[io]) for io in ios]
    ax.legend(handles=handles, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    out = os.path.join(outdir, f"fig_boxplot_{topology}.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Path to exp1_results.csv (raw runs)")
    ap.add_argument("--outdir", default="figs",
                    help="Directory for the figures (default: figs/)")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERRO: {args.csv} não encontrado.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    setup_style()

    runs = load_runs(args.csv)
    rows = derived(aggregate(runs))

    written = []
    for topo in sorted({r["topology"] for r in rows}):
        written.append(fig_exec_vs_nodes(rows, topo, args.outdir))
        written.append(fig_overhead_vs_nodes(rows, topo, args.outdir))
        written.append(fig_weak_efficiency(rows, topo, args.outdir))
        out = fig_save_flush_breakdown(rows, topo, args.outdir)
        if out:
            written.append(out)
        out = fig_boxplot_distribution(runs, topo, args.outdir)
        if out:
            written.append(out)

    for w in written:
        if w:
            print(f"escrito: {w}")


if __name__ == "__main__":
    main()
