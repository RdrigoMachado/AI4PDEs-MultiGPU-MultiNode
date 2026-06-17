#!/usr/bin/env python3
"""Strong scaling: tempo de execução (média +/- desvio padrao) e speedup por
numero de nos, a partir de strong_results.csv.

Le so as linhas SUCESSO. Agrega EXEC_S por NODES: media, desvio padrao (amostral)
e min. Gera dois paineis: (1) tempo medio com barras de erro = std, em log-log,
com a curva ideal; (2) speedup medio vs ideal. Imprime tambem a tabela.

Uso:
    python3 tools/strong_scaling_plot.py
    python3 tools/strong_scaling_plot.py --csv tools/strong_results.csv \
        --out tools/strong_scaling.pdf
"""
import argparse
import csv
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def summarize(times):
    rows = []
    for nodes in sorted(times):
        v = times[nodes]
        mean = statistics.fmean(v)
        std = statistics.stdev(v) if len(v) > 1 else 0.0
        rows.append({
            "nodes": nodes, "n": len(v),
            "mean": mean, "std": std, "min": min(v), "max": max(v),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="tools/strong_results.csv")
    ap.add_argument("--out", default="tools/strong_scaling.pdf")
    args = ap.parse_args()

    rows = summarize(load(args.csv))
    if not rows:
        raise SystemExit("nenhum dado SUCESSO encontrado")

    base = rows[0]            # menor numero de nos = referencia do speedup
    base_nodes, base_mean = base["nodes"], base["mean"]

    print(f"{'nos':>4} {'n':>3} {'media(s)':>9} {'std(s)':>8} "
          f"{'cv%':>6} {'min(s)':>8} {'speedup':>8} {'efic%':>6}")
    for r in rows:
        sp = base_mean / r["mean"] * base_nodes
        eff = sp / r["nodes"] * 100.0
        cv = r["std"] / r["mean"] * 100.0
        print(f"{r['nodes']:>4} {r['n']:>3} {r['mean']:>9.2f} {r['std']:>8.2f} "
              f"{cv:>6.1f} {r['min']:>8.2f} {sp:>8.2f} {eff:>6.1f}")

    nodes = [r["nodes"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]

    fig, ax2 = plt.subplots(1, 1, figsize=(5.5, 4.5))

    # --- Speedup medio +/- std propagado ---
    speedups = [base_mean / m * base_nodes for m in means]
    # propagacao: S = k/m  =>  dS = S * (dm/m)
    sp_err = [s * (sd / m) for s, sd, m in zip(speedups, stds, means)]
    ax2.plot(nodes, nodes, "--", color="gray", label="ideal")
    ax2.errorbar(nodes, speedups, yerr=sp_err, marker="s", capsize=4,
                 color="C1", label="medido (media +/- std)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xticks(nodes)
    ax2.set_xticklabels(nodes)
    ax2.set_yticks(nodes)
    ax2.set_yticklabels(nodes)
    ax2.set_xlabel("numero de nos (4 GPUs/no)")
    ax2.set_ylabel(f"speedup (vs {base_nodes} no)")
    ax2.set_title("Strong scaling - speedup (256x256x12288, 3D, io=none)")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(args.out)
    png = args.out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=150)
    print(f"\nsalvo: {args.out} e {png}")


if __name__ == "__main__":
    main()
