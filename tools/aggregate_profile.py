#!/usr/bin/env python3
"""Agrega CSVs de profiling de múltiplos ranks de um único run.

Uso:
    python aggregate_profile.py <run_dir>
    python aggregate_profile.py /scratch/.../exp0_runs/blocking_r1_<jobid>/

Lê todos os profile_rank*.csv da pasta e produz tabela com mean/std/min/max
de tempo por região, agregando entre ranks. Útil para responder:
  - Quanto do step total é halo? (halo_total / step_total)
  - Quanto do halo é cada eixo? (halo_X / halo_Y / halo_Z)
  - Multigrid domina o forward?
"""

import argparse
import csv
import glob
import os
import statistics
import sys


def load_rank_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def aggregate(run_dir):
    pattern = os.path.join(run_dir, "profile_rank*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERRO: nenhum profile_rank*.csv em {run_dir}", file=sys.stderr)
        return 2

    # region -> list of total_time_s (one per rank)
    times = {}
    counts = {}
    meta = {}
    for fp in files:
        for r in load_rank_csv(fp):
            name = r["region"]
            t = float(r["time_s"])
            c = int(r["count"])
            times.setdefault(name, []).append(t)
            counts.setdefault(name, []).append(c)
            for k in ("topology", "halo_strategy", "world_size", "nx", "ny", "nz"):
                if k in r:
                    meta[k] = r[k]

    # step_total e halo_total dão o denominador de fração.
    step_mean = statistics.mean(times["step_total"]) if "step_total" in times else None
    halo_mean = statistics.mean(times["halo_total"]) if "halo_total" in times else None

    print(f"# Profile aggregate — {run_dir}")
    if meta:
        print(f"# {meta}")
    print(f"# {len(files)} rank(s)")
    print()

    header = f"{'region':<22} {'count':>6} {'mean_s':>10} {'std_s':>9} {'min_s':>10} {'max_s':>10}"
    if step_mean:
        header += f" {'%step':>7}"
    if halo_mean:
        header += f" {'%halo':>7}"
    print(header)
    print("-" * len(header))

    # Ordena por mean_s desc
    region_order = sorted(times.keys(), key=lambda n: -statistics.mean(times[n]))
    for name in region_order:
        ts = times[name]
        cs = counts[name]
        mean_t = statistics.mean(ts)
        std_t = statistics.stdev(ts) if len(ts) > 1 else 0.0
        min_t = min(ts)
        max_t = max(ts)
        # count é o mesmo entre ranks tipicamente; usa o do rank 0
        c = cs[0] if cs else 0

        line = f"{name:<22} {c:>6d} {mean_t:>10.4f} {std_t:>9.4f} {min_t:>10.4f} {max_t:>10.4f}"
        if step_mean and step_mean > 0:
            line += f" {100*mean_t/step_mean:>6.2f}%"
        if halo_mean and halo_mean > 0 and name.startswith("halo"):
            line += f" {100*mean_t/halo_mean:>6.2f}%"
        print(line)

    if step_mean and halo_mean:
        print()
        print(f"halo_total / step_total = {100*halo_mean/step_mean:.2f}%")
        compute_mean = step_mean - halo_mean
        if "io_save" in times:
            io_mean = statistics.mean(times["io_save"])
            print(f"io_save   / step_total = {100*io_mean/step_mean:.2f}%")
            compute_mean -= io_mean
        print(f"compute   / step_total = {100*compute_mean/step_mean:.2f}% (estimado por subtração)")

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Pasta com profile_rank*.csv (WORK_DIR de uma run)")
    args = parser.parse_args()
    if not os.path.isdir(args.run_dir):
        print(f"ERRO: {args.run_dir} não é diretório", file=sys.stderr)
        return 2
    return aggregate(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
