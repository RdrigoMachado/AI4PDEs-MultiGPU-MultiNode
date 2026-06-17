#!/usr/bin/env python3
"""Clean strong-scaling speedup figure for the paper (English, single panel).

Reads strong_results.csv (SUCESSO rows only), reduces EXEC_S per node count by
--metric (default: min, the clean-mode capability, robust to the GPU clock/boost
bimodality), and plots speedup vs ideal. Minimal styling for publication.

Usage:
    python3 tools/strong_scaling_figure.py
    python3 tools/strong_scaling_figure.py --metric median
"""
import argparse
import csv
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REDUCE = {"min": min, "median": statistics.median, "mean": statistics.fmean}


def load(csv_path):
    times = defaultdict(list)
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["STATUS"] != "SUCESSO":
                continue
            try:
                times[int(row["NODES"])].append(float(row["EXEC_S"]))
            except (ValueError, KeyError):
                continue
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="tools/strong_results.csv")
    ap.add_argument("--out", default="tools/strong_scaling_speedup.pdf")
    ap.add_argument("--metric", choices=REDUCE, default="min")
    args = ap.parse_args()

    times = load(args.csv)
    reduce = REDUCE[args.metric]
    nodes = sorted(times)
    t = [reduce(times[n]) for n in nodes]
    base_n, base_t = nodes[0], t[0]
    speedup = [base_t / ti * base_n for ti in t]

    # IQR-based error bars on the speedup (consistent with the paper's
    # median (IQR) reporting). With S = base_t / t, a faster time (q1) maps to a
    # higher speedup and a slower time (q3) to a lower one, so the bars are
    # asymmetric. Only drawn when the central metric is the median.
    yerr = None
    if args.metric == "median":
        q1 = [statistics.quantiles(times[n], n=4)[0] for n in nodes]
        q3 = [statistics.quantiles(times[n], n=4)[2] for n in nodes]
        lo = [s - base_t / hi * base_n for s, hi in zip(speedup, q3)]
        hi = [base_t / lo_ * base_n - s for s, lo_ in zip(speedup, q1)]
        yerr = [lo, hi]

    for i, (n, ti, s) in enumerate(zip(nodes, t, speedup)):
        print(f"{n:>3} nodes  {args.metric}={ti:8.2f}s  speedup={s:6.2f}  "
              f"eff={s / n * 100:5.1f}%")

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot(nodes, nodes, "--", color="0.5", linewidth=1.2, label="Ideal")
    ax.errorbar(nodes, speedup, yerr=yerr, fmt="o-", color="C0", linewidth=1.6,
                markersize=6, capsize=4, label="Measured")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(nodes)
    ax.set_xticklabels(nodes)
    ax.set_yticks(nodes)
    ax.set_yticklabels(nodes)
    ax.set_xlabel("Number of nodes (4 GPUs each)")
    ax.set_ylabel("Speedup (vs. 1 node)")
    ax.grid(True, which="major", ls=":", alpha=0.4)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    fig.savefig(args.out)
    png = args.out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=200)
    print(f"saved: {args.out} and {png}")


if __name__ == "__main__":
    main()
