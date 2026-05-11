#!/usr/bin/env python3
"""Parseia logs de nccl-tests e extrai α (latência) + β (banda) do modelo

    T(B) = α + B / β.

Cada subpasta de <input_dir> deve ter o padrão `<label>_<jobid>/` contendo
`meta.txt`, `sendrecv.txt` e `allreduce.txt` (gerados por
scripts/run_nccl_tests.sbatch).

Faz dois fits por (label, op): um em todos os tamanhos e outro só nos
tamanhos >= --asym-min-bytes (default 1 MiB) para isolar o regime de
banda saturada — α do fit completo é dominado pelo overhead, β do
assintótico reflete a banda efetiva do canal.

Uso:
    python parse_nccl_tests.py data/nccl_runs [--asym-min-bytes 1048576]

Gera dois arquivos no diretório de entrada:
    nccl_fit.csv  — uma linha por (label, op)
    nccl_fit.tex  — tabela LaTeX pronta para a seção 4 do paper
"""

import argparse
import csv
import os
import re
import sys

import numpy as np


RUN_DIR_RE = re.compile(r"^(?P<label>.+)_(?P<jobid>\d+)$")
META_KV_RE = re.compile(r"^(\w+)=(.*)$")
# size count type redop root | OOP{time algbw busbw #wrong} | IP{time algbw busbw #wrong}
ROW_RE = re.compile(
    r"^\s*(\d+)\s+\d+\s+\w+\s+\w+\s+-?\d+"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\S+"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s*$"
)


def load_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            m = META_KV_RE.match(line.strip())
            if m:
                meta[m.group(1)] = m.group(2)
    return meta


def load_table(path):
    """Retorna array (N,3): size_bytes, time_us_ip, busbw_ip_GBs."""
    rows = []
    with open(path) as f:
        for line in f:
            m = ROW_RE.match(line)
            if not m:
                continue
            size = int(m.group(1))
            time_ip_us = float(m.group(5))
            busbw_ip = float(m.group(7))
            rows.append((size, time_ip_us, busbw_ip))
    if not rows:
        raise ValueError(f"sem linhas de medição em {path}")
    return np.array(rows, dtype=float)


def fit(sizes_bytes, times_us):
    """Fit linear t_us = α_us + B / β_bytes_per_us → retorna (α_us, β_GBs)."""
    if len(sizes_bytes) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(sizes_bytes, times_us, 1)
    if slope <= 0:
        return intercept, float("inf")
    beta_bytes_per_us = 1.0 / slope
    beta_GBs = beta_bytes_per_us * 1e-3  # B/μs → GB/s
    return intercept, beta_GBs


def analyze_op(table, asym_min):
    sizes, times, busbw = table[:, 0], table[:, 1], table[:, 2]
    a_full, b_full = fit(sizes, times)
    mask = sizes >= asym_min
    if mask.sum() >= 2:
        a_asym, b_asym = fit(sizes[mask], times[mask])
    else:
        a_asym, b_asym = float("nan"), float("nan")
    return {
        "alpha_us": a_full,
        "beta_GBs": b_full,
        "alpha_us_asym": a_asym,
        "beta_GBs_asym": b_asym,
        "min_time_us": float(times.min()),
        "peak_busbw_GBs": float(busbw.max()),
        "n_points": int(len(sizes)),
        "n_asym": int(mask.sum()),
    }


def collect(input_dir, asym_min):
    results = []
    for entry in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, entry)
        if not os.path.isdir(path):
            continue
        m = RUN_DIR_RE.match(entry)
        if not m:
            continue
        label, jobid = m.group("label"), m.group("jobid")
        meta = load_meta(os.path.join(path, "meta.txt"))
        for op in ("sendrecv", "allreduce"):
            log = os.path.join(path, f"{op}.txt")
            if not os.path.isfile(log):
                print(f"[skip] {entry}/{op}.txt ausente", file=sys.stderr)
                continue
            try:
                table = load_table(log)
            except ValueError as e:
                print(f"[skip] {entry}/{op}.txt: {e}", file=sys.stderr)
                continue
            stats = analyze_op(table, asym_min)
            results.append({
                "label": label,
                "jobid": jobid,
                "nodes": int(meta.get("nodes", 0)),
                "tasks_per_node": int(meta.get("tasks_per_node", 0)),
                "total_tasks": int(meta.get("total_tasks", 0)),
                "op": op,
                **stats,
            })
    return results


def write_csv(results, path):
    if not results:
        return
    fields = list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)


def write_tex(results, path):
    """Tabela LaTeX só com sendrecv (regime do halo)."""
    rows = [r for r in results if r["op"] == "sendrecv"]
    rows.sort(key=lambda r: (r["nodes"], r["tasks_per_node"]))
    with open(path, "w") as f:
        f.write("% gerado por tools/parse_nccl_tests.py\n")
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        f.write("Config & Nodes & GPUs/n & Ranks "
                "& $\\alpha$ ($\\mu$s) & $\\beta$ (GB/s) "
                "& $\\alpha_{\\text{asym}}$ & $\\beta_{\\text{asym}}$ \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r['label'].replace('_', '\\_')} & "
                f"{r['nodes']} & {r['tasks_per_node']} & {r['total_tasks']} & "
                f"{r['alpha_us']:.2f} & {r['beta_GBs']:.2f} & "
                f"{r['alpha_us_asym']:.2f} & {r['beta_GBs_asym']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")


def print_summary(results):
    width = max(len(r["label"]) for r in results)
    hdr = (f"{'label':<{width}}  {'op':<9}  {'ranks':>5}  "
           f"{'α(μs)':>7}  {'β(GB/s)':>8}  {'α_asym':>7}  {'β_asym':>8}  "
           f"{'min_us':>7}  {'peak_bw':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['label']:<{width}}  {r['op']:<9}  {r['total_tasks']:>5}  "
              f"{r['alpha_us']:>7.2f}  {r['beta_GBs']:>8.2f}  "
              f"{r['alpha_us_asym']:>7.2f}  {r['beta_GBs_asym']:>8.2f}  "
              f"{r['min_time_us']:>7.2f}  {r['peak_busbw_GBs']:>8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="dir contendo subpastas <label>_<jobid>/")
    ap.add_argument("--asym-min-bytes", type=int, default=1 << 20,
                    help="tamanho mínimo (bytes) para fit assintótico (default 1 MiB)")
    args = ap.parse_args()

    results = collect(args.input_dir, args.asym_min_bytes)
    if not results:
        print("nenhum resultado encontrado", file=sys.stderr)
        sys.exit(1)
    print_summary(results)
    write_csv(results, os.path.join(args.input_dir, "nccl_fit.csv"))
    write_tex(results, os.path.join(args.input_dir, "nccl_fit.tex"))
    print(f"\nescrito: {args.input_dir}/nccl_fit.csv")
    print(f"escrito: {args.input_dir}/nccl_fit.tex")


if __name__ == "__main__":
    main()
