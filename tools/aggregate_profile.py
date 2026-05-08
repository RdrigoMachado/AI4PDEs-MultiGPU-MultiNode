#!/usr/bin/env python3
"""Agrega CSVs de profiling.

Dois modos, detectados pelo conteúdo do diretório:

(A) Single-run: <run_dir> contém profile_rank*.csv direto.
    Agrega entre ranks → tabela região × tempo (count, mean, std, min, max).
    Também imprime métricas pontuais (memória etc.) se metrics_rank*.csv existir.

(B) Multi-run: <parent_dir> contém subpastas no padrão
    prof_<topo>_<NX>x<NY>x<NZ>_<NODES>n_r<RUN>_<jobid>/
    com profile_rank*.csv dentro. Agrega inter-rank por run, depois inter-run
    por config (descartando r0 — warmup). Gera dois CSVs no parent_dir:
      profile_per_run.csv     — uma linha por (config, run, region)
      profile_per_config.csv  — uma linha por (config, region) com inter-run
    e imprime resumo por config no stdout.

Uso:
    python aggregate_profile.py <dir>
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys


RUN_DIR_RE = re.compile(
    r"^prof_(?P<topo>[^_]+)_(?P<nx>\d+)x(?P<ny>\d+)x(?P<nz>\d+)"
    r"_(?P<nodes>\d+)n_r(?P<run>\d+)_(?P<jobid>\d+)$"
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_rank_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_run_dir_name(name):
    m = RUN_DIR_RE.match(name)
    if not m:
        return None
    return {
        "topology": m["topo"],
        "nx": int(m["nx"]),
        "ny": int(m["ny"]),
        "nz": int(m["nz"]),
        "nodes": int(m["nodes"]),
        "run": int(m["run"]),
        "jobid": int(m["jobid"]),
    }


def has_profile_csvs_direct(d):
    return bool(glob.glob(os.path.join(d, "profile_rank*.csv")))


# ---------------------------------------------------------------------------
# Modo A: single-run aggregate (compatível com versão anterior)
# ---------------------------------------------------------------------------
def aggregate_single_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "profile_rank*.csv")))
    if not files:
        print(f"ERRO: nenhum profile_rank*.csv em {run_dir}", file=sys.stderr)
        return 2

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

    region_order = sorted(times.keys(), key=lambda n: -statistics.mean(times[n]))
    for name in region_order:
        ts = times[name]
        cs = counts[name]
        mean_t = statistics.mean(ts)
        std_t = statistics.stdev(ts) if len(ts) > 1 else 0.0
        min_t = min(ts)
        max_t = max(ts)
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

    # Métricas pontuais (memória, etc.)
    metrics_files = sorted(glob.glob(os.path.join(run_dir, "metrics_rank*.csv")))
    if metrics_files:
        print()
        print(f"# Métricas pontuais — {len(metrics_files)} rank(s)")
        per_metric = {}
        for fp in metrics_files:
            for r in load_rank_csv(fp):
                per_metric.setdefault(r["metric"], []).append(float(r["value"]))
        print(f"{'metric':<28} {'mean':>10} {'min':>10} {'max':>10}")
        print("-" * 62)
        for name in sorted(per_metric):
            vs = per_metric[name]
            print(f"{name:<28} {statistics.mean(vs):>10.4f} {min(vs):>10.4f} {max(vs):>10.4f}")

    return 0


# ---------------------------------------------------------------------------
# Modo B: multi-run aggregate
# ---------------------------------------------------------------------------
def collect_runs(parent_dir):
    """Retorna lista de dicts com {meta, run_dir} para cada subpasta válida."""
    runs = []
    for entry in sorted(os.listdir(parent_dir)):
        p = os.path.join(parent_dir, entry)
        if not os.path.isdir(p):
            continue
        meta = parse_run_dir_name(entry)
        if meta is None:
            continue
        if not has_profile_csvs_direct(p):
            continue
        runs.append({"meta": meta, "run_dir": p})
    return runs


def aggregate_run_inter_rank(run_dir):
    """Para um run, retorna {region: total_time_s} médio entre ranks."""
    files = sorted(glob.glob(os.path.join(run_dir, "profile_rank*.csv")))
    times_per_region = {}
    counts_per_region = {}
    for fp in files:
        for r in load_rank_csv(fp):
            name = r["region"]
            times_per_region.setdefault(name, []).append(float(r["time_s"]))
            counts_per_region.setdefault(name, []).append(int(r["count"]))
    out = {}
    for name, vs in times_per_region.items():
        out[name] = {
            "mean_rank": statistics.mean(vs),
            "max_rank": max(vs),
            "min_rank": min(vs),
            "count": counts_per_region[name][0] if counts_per_region[name] else 0,
        }
    return out


def aggregate_run_metrics(run_dir):
    """Métricas pontuais agregadas inter-rank: max (pico) faz sentido para mem."""
    files = sorted(glob.glob(os.path.join(run_dir, "metrics_rank*.csv")))
    out = {}
    per_metric = {}
    for fp in files:
        for r in load_rank_csv(fp):
            per_metric.setdefault(r["metric"], []).append(float(r["value"]))
    for name, vs in per_metric.items():
        out[name] = {
            "mean": statistics.mean(vs),
            "max": max(vs),
            "min": min(vs),
        }
    return out


def cfg_key(meta):
    return f"{meta['topology']}_{meta['nx']}x{meta['ny']}x{meta['nz']}_{meta['nodes']}n"


def aggregate_multi_run(parent_dir, drop_warmup=True):
    runs = collect_runs(parent_dir)
    if not runs:
        print(f"ERRO: nenhuma subpasta prof_*_r*_<jobid> em {parent_dir}", file=sys.stderr)
        return 2

    # Estrutura: per_cfg[cfg][region] = lista de mean_rank (1 por run medido)
    per_cfg_time = {}
    per_cfg_count = {}
    per_cfg_metrics = {}  # cfg -> metric -> list of (max-rank-by-run)
    per_run_rows = []     # linhas de profile_per_run.csv

    n_warmup_skipped = 0
    for entry in runs:
        meta = entry["meta"]
        run_dir = entry["run_dir"]
        cfg = cfg_key(meta)
        agg_t = aggregate_run_inter_rank(run_dir)
        agg_m = aggregate_run_metrics(run_dir)

        is_warmup = (meta["run"] == 0)
        for region, d in agg_t.items():
            per_run_rows.append({
                "config": cfg,
                "topology": meta["topology"],
                "nx": meta["nx"],
                "ny": meta["ny"],
                "nz": meta["nz"],
                "nodes": meta["nodes"],
                "run": meta["run"],
                "jobid": meta["jobid"],
                "warmup": int(is_warmup),
                "region": region,
                "count": d["count"],
                "mean_rank_s": d["mean_rank"],
                "max_rank_s": d["max_rank"],
                "min_rank_s": d["min_rank"],
            })

        if drop_warmup and is_warmup:
            n_warmup_skipped += 1
            continue

        for region, d in agg_t.items():
            per_cfg_time.setdefault(cfg, {}).setdefault(region, []).append(d["mean_rank"])
            per_cfg_count.setdefault(cfg, {}).setdefault(region, d["count"])
        for metric, d in agg_m.items():
            per_cfg_metrics.setdefault(cfg, {}).setdefault(metric, []).append(d["max"])

    # Dump per_run
    per_run_path = os.path.join(parent_dir, "profile_per_run.csv")
    with open(per_run_path, "w", newline="") as f:
        cols = ["config", "topology", "nx", "ny", "nz", "nodes", "run", "jobid",
                "warmup", "region", "count", "mean_rank_s", "max_rank_s", "min_rank_s"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in per_run_rows:
            w.writerow(row)
    print(f"Wrote {per_run_path}  ({len(per_run_rows)} rows)")

    # Dump per_config (timing)
    per_cfg_path = os.path.join(parent_dir, "profile_per_config.csv")
    with open(per_cfg_path, "w", newline="") as f:
        cols = ["config", "region", "n_runs", "count_per_step",
                "mean_s", "std_s", "min_s", "max_s"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for cfg, region_dict in per_cfg_time.items():
            for region, vs in region_dict.items():
                m = statistics.mean(vs)
                sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
                w.writerow({
                    "config": cfg,
                    "region": region,
                    "n_runs": len(vs),
                    "count_per_step": per_cfg_count[cfg][region],
                    "mean_s": f"{m:.6f}",
                    "std_s": f"{sd:.6f}",
                    "min_s": f"{min(vs):.6f}",
                    "max_s": f"{max(vs):.6f}",
                })
    print(f"Wrote {per_cfg_path}")

    # Dump per_config metrics (memória etc.)
    per_cfg_met_path = os.path.join(parent_dir, "metrics_per_config.csv")
    with open(per_cfg_met_path, "w", newline="") as f:
        cols = ["config", "metric", "n_runs", "mean", "min", "max"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for cfg, met_dict in per_cfg_metrics.items():
            for metric, vs in met_dict.items():
                w.writerow({
                    "config": cfg,
                    "metric": metric,
                    "n_runs": len(vs),
                    "mean": f"{statistics.mean(vs):.6f}",
                    "min": f"{min(vs):.6f}",
                    "max": f"{max(vs):.6f}",
                })
    print(f"Wrote {per_cfg_met_path}")

    # Resumo no stdout
    print()
    print(f"# Multi-run aggregate — {parent_dir}")
    print(f"# {len(runs)} run dir(s); warmup descartado: {n_warmup_skipped}")
    print()
    for cfg in sorted(per_cfg_time):
        print(f"=== {cfg} ===")
        regions = per_cfg_time[cfg]
        step_mean = statistics.mean(regions["step_total"]) if "step_total" in regions else None
        halo_mean = statistics.mean(regions["halo_total"]) if "halo_total" in regions else None
        order = sorted(regions, key=lambda n: -statistics.mean(regions[n]))
        header = f"  {'region':<22} {'mean_s':>10} {'std_s':>9}"
        if step_mean:
            header += f" {'%step':>7}"
        print(header)
        for region in order:
            vs = regions[region]
            m = statistics.mean(vs)
            sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
            line = f"  {region:<22} {m:>10.4f} {sd:>9.4f}"
            if step_mean and step_mean > 0:
                line += f" {100*m/step_mean:>6.2f}%"
            print(line)
        if cfg in per_cfg_metrics:
            print(f"  {'-- metrics --'}")
            for metric, vs in per_cfg_metrics[cfg].items():
                print(f"  {metric:<22} {statistics.mean(vs):>10.4f}")
        print()

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Pasta única (single-run) ou pasta-pai (multi-run)")
    parser.add_argument("--keep-warmup", action="store_true",
                        help="Não descartar r0 do agregado per_config (modo multi-run)")
    args = parser.parse_args()
    if not os.path.isdir(args.path):
        print(f"ERRO: {args.path} não é diretório", file=sys.stderr)
        return 2
    if has_profile_csvs_direct(args.path):
        return aggregate_single_run(args.path)
    return aggregate_multi_run(args.path, drop_warmup=not args.keep_warmup)


if __name__ == "__main__":
    sys.exit(main())
