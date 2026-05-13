#!/usr/bin/env python3
"""Aggregate Exp 1 results: mean / std / min / max per (topology, io_mode, nodes).

Reads exp1_results.csv produced by run_exp1.sbatch and emits:
  - <input_dir>/exp1_summary.csv : one row per cell with statistics
  - stdout                       : formatted table for quick inspection

Usage:
    python tools/exp1_summary.py path/to/exp1_results.csv
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


def load_runs(path):
    """Return list of SUCESSO rows as dicts, with numeric fields parsed."""
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


def stats(values):
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {"n": n, "mean": mean, "std": std,
            "min": min(values), "max": max(values)}


def aggregate(runs):
    """Group by (topology, io_mode, nodes) and compute stats."""
    groups = defaultdict(list)
    for r in runs:
        key = (r["topology"], r["io_mode"], r["nodes"])
        groups[key].append(r)

    rows = []
    for (topo, io, nodes), items in sorted(groups.items(),
                                            key=lambda x: (x[0][0], x[0][2], x[0][1])):
        exec_stats = stats([r["exec_s"] for r in items])
        save_stats = stats([r["save_s"] for r in items])
        flush_stats = stats([r["flush_s"] for r in items])
        rows.append({
            "topology": topo, "io_mode": io, "nodes": nodes,
            "n_runs": exec_stats["n"],
            "exec_mean": exec_stats["mean"], "exec_std": exec_stats["std"],
            "exec_min": exec_stats["min"], "exec_max": exec_stats["max"],
            "save_mean": save_stats["mean"], "save_std": save_stats["std"],
            "flush_mean": flush_stats["mean"], "flush_std": flush_stats["std"],
        })
    return rows


def io_overhead(rows):
    """Compute (exec_mean / exec_none - 1) × 100 for naive and async."""
    baseline = {(r["topology"], r["nodes"]): r["exec_mean"]
                for r in rows if r["io_mode"] == "none"}
    for r in rows:
        key = (r["topology"], r["nodes"])
        if key in baseline and r["io_mode"] != "none":
            r["overhead_pct"] = (r["exec_mean"] / baseline[key] - 1) * 100
        else:
            r["overhead_pct"] = None
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fields = ["topology", "io_mode", "nodes", "n_runs",
              "exec_mean", "exec_std", "exec_min", "exec_max",
              "save_mean", "save_std", "flush_mean", "flush_std",
              "overhead_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fields}
            w.writerow(row)


def print_table(rows):
    header = (f"{'topo':<5}  {'io_mode':<6}  {'nodes':>5}  {'n':>3}  "
              f"{'exec_mean':>10}  {'exec_std':>8}  "
              f"{'save':>6}  {'flush':>6}  {'overhead':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        ov = f"{r['overhead_pct']:+6.1f}%" if r["overhead_pct"] is not None else "    —  "
        print(f"{r['topology']:<5}  {r['io_mode']:<6}  {r['nodes']:>5}  "
              f"{r['n_runs']:>3}  {r['exec_mean']:>10.2f}  {r['exec_std']:>8.2f}  "
              f"{r['save_mean']:>6.2f}  {r['flush_mean']:>6.2f}  {ov:>8}")


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

    rows = io_overhead(aggregate(runs))
    print_table(rows)

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                            "exp1_summary.csv")
    write_csv(rows, out_path)
    print(f"\nescrito: {out_path}")


if __name__ == "__main__":
    main()
