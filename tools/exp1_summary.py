#!/usr/bin/env python3
"""Aggregate Exp 1 results and emit summary CSV + Markdown report.

Reads exp1_results.csv produced by run_exp1.sbatch and writes:
  - <input_dir>/exp1_summary.csv : one row per cell with robust statistics
  - <input_dir>/exp1_report.md   : human-readable analysis for the paper
  - stdout                       : compact table for quick inspection

The median + IQR are reported as primary metrics — Lustre and queue
contention generate long-tailed run-time distributions where mean +
std would either overstate the overhead or invent speedups > 1.

Usage:
    python tools/exp1_summary.py path/to/exp1_results.csv
"""

import argparse
import csv
import datetime as dt
import math
import os
import statistics
import sys
from collections import defaultdict


def load_runs(path):
    """Return SUCESSO rows as dicts with numeric fields parsed."""
    runs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("STATUS") != "SUCESSO":
                continue
            try:
                runs.append({
                    "topology": row["TOPOLOGY"],
                    "io_mode": row["IO_MODE"],
                    "nodes": int(row["NODES"]),
                    "run_id": int(row["RUN_ID"]),
                    "exec_s": float(row["EXEC_S"]),
                    "save_s": float(row["SAVE_S"] or 0),
                    "flush_s": float(row["FLUSH_S"] or 0),
                })
            except (ValueError, KeyError):
                continue
    return runs


def percentile(values, p):
    """Linear-interpolation percentile (numpy-compatible)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return s[int(idx)]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def cell_stats(values):
    if not values:
        return {"n": 0}
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    q1 = percentile(values, 25)
    median = percentile(values, 50)
    q3 = percentile(values, 75)
    iqr = q3 - q1
    fence_lo, fence_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = sorted([v for v in values if v < fence_lo or v > fence_hi])
    return {
        "n": n, "mean": mean, "std": std,
        "median": median, "q1": q1, "q3": q3, "iqr": iqr,
        "min": min(values), "max": max(values),
        "n_outliers": len(outliers),
    }


def aggregate(runs):
    """Group by (topology, io_mode, nodes) → stats over exec/save/flush."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["topology"], r["io_mode"], r["nodes"])].append(r)

    rows = []
    for (topo, io, nodes), items in groups.items():
        exec_v = [r["exec_s"] for r in items]
        save_v = [r["save_s"] for r in items]
        flush_v = [r["flush_s"] for r in items]
        exec_s = cell_stats(exec_v)
        save_s = cell_stats(save_v)
        flush_s = cell_stats(flush_v)
        rows.append({
            "topology": topo, "io_mode": io, "nodes": nodes,
            "n_runs": exec_s["n"],
            # exec
            "exec_median": exec_s["median"], "exec_q1": exec_s["q1"], "exec_q3": exec_s["q3"],
            "exec_iqr": exec_s["iqr"], "exec_mean": exec_s["mean"], "exec_std": exec_s["std"],
            "exec_min": exec_s["min"], "exec_max": exec_s["max"],
            "exec_outliers": exec_s["n_outliers"],
            # save / flush
            "save_median": save_s["median"], "save_iqr": save_s["iqr"],
            "flush_median": flush_s["median"], "flush_iqr": flush_s["iqr"],
        })
    rows.sort(key=lambda r: (r["topology"], r["nodes"], r["io_mode"]))
    return rows


def derived(rows):
    """Add overhead vs baseline ('none' at same N) and weak efficiency."""
    baseline_exec = {(r["topology"], r["nodes"]): r["exec_median"]
                     for r in rows if r["io_mode"] == "none"}
    baseline_1n = {r["topology"]: r["exec_median"]
                   for r in rows if r["io_mode"] == "none" and r["nodes"] == 1}

    for r in rows:
        b = baseline_exec.get((r["topology"], r["nodes"]))
        r["overhead_pct"] = (r["exec_median"] / b - 1) * 100 if b else None
        b1 = baseline_1n.get(r["topology"])
        r["weak_efficiency"] = b1 / r["exec_median"] if b1 else None
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(rows, path):
    if not rows:
        return
    fields = [
        "topology", "io_mode", "nodes", "n_runs",
        "exec_median", "exec_q1", "exec_q3", "exec_iqr",
        "exec_mean", "exec_std", "exec_min", "exec_max", "exec_outliers",
        "save_median", "save_iqr", "flush_median", "flush_iqr",
        "overhead_pct", "weak_efficiency",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def fmt(v, spec="6.2f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  —  "
    return format(v, spec)


def print_table(rows):
    print(f"{'topo':<5}  {'io':<6}  {'N':>3}  {'n':>3}  "
          f"{'median':>7}  {'IQR':>6}  {'min':>7}  {'max':>7}  "
          f"{'over%':>7}  {'eff':>5}  {'out':>3}")
    print("-" * 78)
    for r in rows:
        print(f"{r['topology']:<5}  {r['io_mode']:<6}  {r['nodes']:>3}  "
              f"{r['n_runs']:>3}  "
              f"{fmt(r['exec_median'], '7.2f')}  {fmt(r['exec_iqr'], '6.2f')}  "
              f"{fmt(r['exec_min'], '7.2f')}  {fmt(r['exec_max'], '7.2f')}  "
              f"{fmt(r['overhead_pct'], '+6.1f')}%  "
              f"{fmt(r['weak_efficiency'], '5.3f')}  "
              f"{r['exec_outliers']:>3}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def md_table(headers, rows, aligns=None):
    if aligns is None:
        aligns = [":---:"] * len(headers)
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(aligns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def write_report(rows, runs, csv_path, out_path):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_total = len(runs)
    topologies = sorted({r["topology"] for r in rows})

    lines = []
    lines.append("# Exp 1 — Weak Scaling with I/O: Analysis Report")
    lines.append("")
    lines.append(f"Generated: {now}  ")
    lines.append(f"Input: `{csv_path}`  ")
    lines.append(f"Total successful runs aggregated: **{n_total}**")
    lines.append("")
    lines.append("Methodology: per (topology, io_mode, nodes) cell, "
                 "**median + IQR** are reported as primary statistics — "
                 "Lustre and queue contention generate long-tailed "
                 "distributions where mean ± std would either overstate "
                 "the I/O cost or invent speedups > 1. Mean / std are kept "
                 "in the CSV for reference.")
    lines.append("")
    lines.append("Outliers are counted using Tukey's fence: values "
                 "outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR].")
    lines.append("")

    # Baseline section
    for topo in topologies:
        none_rows = [r for r in rows if r["topology"] == topo and r["io_mode"] == "none"]
        if not none_rows:
            continue
        lines.append(f"## Baseline (no I/O) — {topo.upper()}")
        lines.append("")
        lines.append("Weak-scaling reference. Without I/O, the solver "
                     "should scale ideally (efficiency ≈ 1.0).")
        lines.append("")
        body = [[str(r["nodes"]),
                 f"{r['exec_median']:.2f}",
                 f"{r['exec_iqr']:.2f}",
                 f"{r['exec_std']:.2f}",
                 f"{r['weak_efficiency']:.3f}" if r["weak_efficiency"] is not None else "—",
                 str(r["n_runs"])]
                for r in sorted(none_rows, key=lambda r: r["nodes"])]
        lines.append(md_table(
            ["Nodes", "Median exec (s)", "IQR (s)", "Std (s)", "Weak eff.", "n"],
            body,
            ["---:", "---:", "---:", "---:", "---:", "---:"]))
        lines.append("")

    # Per-topology I/O comparison
    for topo in topologies:
        sub = [r for r in rows if r["topology"] == topo]
        nodes_set = sorted({r["nodes"] for r in sub})
        lines.append(f"## I/O cost vs baseline — {topo.upper()}")
        lines.append("")
        lines.append("Each row compares one I/O mode against the no-I/O "
                     "baseline at the same node count. `overhead = "
                     "(T_iomode − T_none) / T_none × 100%`. Negative "
                     "overheads indicate the I/O mode ran slightly faster "
                     "than the baseline median — pure noise within IQR.")
        lines.append("")
        body = []
        for n in nodes_set:
            base = next((r for r in sub
                         if r["nodes"] == n and r["io_mode"] == "none"), None)
            for io in ("naive", "async"):
                rr = next((r for r in sub
                           if r["nodes"] == n and r["io_mode"] == io), None)
                if not rr or not base:
                    continue
                body.append([
                    str(n), io,
                    f"{base['exec_median']:.2f}",
                    f"{rr['exec_median']:.2f}",
                    f"{rr['exec_iqr']:.2f}",
                    f"{rr['overhead_pct']:+.1f}%",
                    f"{rr['weak_efficiency']:.3f}",
                    f"{rr['save_median']:.2f}",
                    f"{rr['flush_median']:.2f}",
                    str(rr["exec_outliers"]),
                ])
        lines.append(md_table(
            ["Nodes", "I/O mode", "Baseline med (s)", "Median (s)",
             "IQR (s)", "Overhead", "Weak eff.", "Save med (s)",
             "Flush med (s)", "Outliers"],
            body,
            ["---:", ":---:", "---:", "---:", "---:", "---:", "---:",
             "---:", "---:", "---:"]))
        lines.append("")

    # Highlights / narrative
    lines.append("## Observations")
    lines.append("")

    # Compute headline numbers automatically (3D if present, else first topo)
    head_topo = "3d" if "3d" in topologies else topologies[0]
    head_rows = [r for r in rows if r["topology"] == head_topo]
    none_by_n = {r["nodes"]: r for r in head_rows if r["io_mode"] == "none"}
    naive_by_n = {r["nodes"]: r for r in head_rows if r["io_mode"] == "naive"}
    async_by_n = {r["nodes"]: r for r in head_rows if r["io_mode"] == "async"}

    if none_by_n:
        max_n = max(none_by_n)
        lines.append(f"- **Baseline weak scaling**: in {head_topo.upper()}, "
                     f"the no-I/O median goes from "
                     f"{none_by_n[min(none_by_n)]['exec_median']:.1f}s "
                     f"at 1 node to {none_by_n[max_n]['exec_median']:.1f}s "
                     f"at {max_n} nodes — efficiency ≈ "
                     f"{none_by_n[max_n]['weak_efficiency']:.3f}.")
    if naive_by_n and async_by_n:
        max_n = max(set(naive_by_n) & set(async_by_n))
        n = naive_by_n[max_n]
        a = async_by_n[max_n]
        lines.append(f"- **At {max_n} nodes**, naive I/O adds "
                     f"{n['overhead_pct']:+.1f}% to the baseline median, "
                     f"async only {a['overhead_pct']:+.1f}%. "
                     f"The async over naive ratio "
                     f"({n['exec_median'] / a['exec_median']:.2f}×) is the "
                     f"headline speedup of overlapping I/O at this scale.")
    cells_with_outliers = [r for r in rows if r["exec_outliers"] > 0]
    if cells_with_outliers:
        lines.append("- **Cells flagged with outliers** (Lustre contention "
                     "or transient queue effects):")
        for r in cells_with_outliers:
            lines.append(f"    - {r['topology']} / {r['io_mode']} / "
                         f"{r['nodes']}n: {r['exec_outliers']} outliers "
                         f"(IQR={r['exec_iqr']:.1f}s)")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Path to exp1_results.csv")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERRO: {args.csv} não encontrado.", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(args.csv)
    if not runs:
        print("Nenhuma linha SUCESSO encontrada.", file=sys.stderr)
        sys.exit(1)

    rows = derived(aggregate(runs))
    print_table(rows)

    out_dir = os.path.dirname(os.path.abspath(args.csv))
    csv_out = os.path.join(out_dir, "exp1_summary.csv")
    md_out = os.path.join(out_dir, "exp1_report.md")
    write_csv(rows, csv_out)
    write_report(rows, runs, args.csv, md_out)
    print(f"\nescrito: {csv_out}")
    print(f"escrito: {md_out}")


if __name__ == "__main__":
    main()
