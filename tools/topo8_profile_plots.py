#!/usr/bin/env python3
"""Topology comparison plots (@8 nós, grid global FIXA): tempo + profiling.

Lê os CSVs por rank gravados pelo run_profile.sbatch (profile_rank*.csv dentro de
cada prof_<topo>_<grid>_<N>n_r<run>_<jobid>/) e produz duas figuras:

  topo8_time.{pdf,png}     tempo de execução (step_total) por topologia; barra =
                           média entre runs, com os runs individuais sobrepostos.
  topo8_profile.{pdf,png}  (esq.) decomposição por estágio em compute vs halo,
                           3d vs slab-y-2d; (dir.) halo por eixo (X/Y/Z) — a
                           "borda entre GPUs" que domina a comunicação.

Compara MESMA grid: cada decomposição fatia do seu jeito, então o nlevel difere
(3d=6, slab-y-2d=5) — fato reportado, não corrigido. Agrega entre ranks pela
média (GPU típica); step_total do rank mais lento ≈ Execution_time do .out.

Uso:
    python3 tools/topo8_profile_plots.py --root ~/inpe/profiles
    python3 tools/topo8_profile_plots.py --root <dir> --grid 3200x1280x1280 --nodes 8
"""
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

STAGES = ["predictor", "corrector", "pre_pressure", "multigrid", "post_pressure"]
STAGE_LABEL = {
    "predictor": "predictor", "corrector": "corrector",
    "pre_pressure": "pre_pres.", "multigrid": "multigrid",
    "post_pressure": "post_pres.",
}
TOPO_ORDER = ["3d", "slab-y-2d"]
TOPO_COLOR = {"3d": "#4C72B0", "slab-y-2d": "#DD8452"}
TOPO_LABEL = {"3d": "3D  (corta Z / IB)", "slab-y-2d": "slab-y-2d  (corta Y / IB)"}

DIR_RE = re.compile(r"prof_(?P<topo>.+)_(?P<grid>\d+x\d+x\d+)_(?P<nodes>\d+)n_r(?P<run>\d+)_")


def load_run(run_dir):
    """{region: mean_time_s_entre_ranks}, e o step_total máximo (caminho crítico)."""
    per_region = defaultdict(list)
    for fp in glob.glob(os.path.join(run_dir, "profile_rank*.csv")):
        with open(fp, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    per_region[row["region"]].append(float(row["time_s"]))
                except (ValueError, KeyError):
                    pass
    if not per_region:
        return None
    region_mean = {r: statistics.fmean(v) for r, v in per_region.items()}
    step_max = max(per_region.get("step_total", [region_mean.get("step_total", 0.0)]))
    return region_mean, step_max


def collect(root, grid, nodes):
    """runs[topo] = lista de (region_mean, step_max), filtrando grid/nodes."""
    runs = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "**", "prof_*"), recursive=True)):
        if not os.path.isdir(d):
            continue
        m = DIR_RE.search(os.path.basename(d))
        if not m:
            continue
        if grid and m["grid"] != grid:
            continue
        if nodes and int(m["nodes"]) != nodes:
            continue
        res = load_run(d)
        if res:
            runs[m["topo"]].append(res)
    return runs


def ordered_topos(runs):
    return [t for t in TOPO_ORDER if t in runs] + [t for t in runs if t not in TOPO_ORDER]


def avg(runs_list, region):
    """Média, entre runs, do valor médio-entre-ranks de `region`."""
    vals = [rm.get(region, 0.0) for rm, _ in runs_list]
    return statistics.fmean(vals) if vals else 0.0


def plot_time(runs, out):
    topos = ordered_topos(runs)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for i, topo in enumerate(topos):
        steps = [smax for _, smax in runs[topo]]
        mean = statistics.fmean(steps)
        lo, hi = min(steps), max(steps)
        ax.bar(i, mean, width=0.6, color=TOPO_COLOR.get(topo, f"C{i}"), alpha=0.9,
               yerr=[[mean - lo], [hi - mean]] if len(steps) > 1 else None, capsize=5)
        ax.scatter([i] * len(steps), steps, color="0.15", s=22, zorder=3)
        ax.text(i, mean, f"{mean:.1f} s", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(range(len(topos)))
    ax.set_xticklabels([TOPO_LABEL.get(t, t) for t in topos], fontsize=9)
    ax.set_ylabel("Execution time — step_total (s)")
    ax.set_title(f"Same global grid, {max(len(runs[t]) for t in topos)} run(s)/topo")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.margins(y=0.15)
    fig.tight_layout()
    _save(fig, out)


def plot_profile(runs, out):
    topos = ordered_topos(runs)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # --- Painel A: por estágio, compute vs halo, agrupado por topologia ---
    x = range(len(STAGES))
    n = len(topos)
    width = 0.8 / n
    for j, topo in enumerate(topos):
        rl = runs[topo]
        off = (j - (n - 1) / 2) * width
        comp, halo = [], []
        for s in STAGES:
            total = avg(rl, s)
            h = avg(rl, f"halo_in_{s}")
            halo.append(h)
            comp.append(max(total - h, 0.0))
        xs = [xi + off for xi in x]
        c = TOPO_COLOR.get(topo, f"C{j}")
        axA.bar(xs, comp, width=width, color=c, alpha=0.9)
        axA.bar(xs, halo, width=width, bottom=comp, color=c, alpha=0.9,
                hatch="////", edgecolor="white", linewidth=0.0)
    axA.set_xticks(list(x))
    axA.set_xticklabels([STAGE_LABEL[s] for s in STAGES], rotation=20, ha="right", fontsize=9)
    axA.set_ylabel("Tempo acumulado, 40 passos (s)")
    axA.set_title("Por estágio: compute (sólido) vs halo (hachurado)")
    axA.grid(True, axis="y", ls=":", alpha=0.4)
    legend = [Patch(facecolor=TOPO_COLOR.get(t, f"C{i}"), label=TOPO_LABEL.get(t, t))
              for i, t in enumerate(topos)]
    legend.append(Patch(facecolor="0.6", hatch="////", edgecolor="white", label="halo (comunicação)"))
    axA.legend(handles=legend, frameon=False, fontsize=8, loc="upper left")

    # --- Painel B: halo por eixo (a "borda entre GPUs") ---
    def halo_axis(rl):
        return {
            "X": avg(rl, "halo_X"),
            "Y": avg(rl, "halo_Y"),
            "Z": avg(rl, "halo_Z_phase1") + avg(rl, "halo_Z_phase2"),
        }
    axes_lbl = ["X", "Y", "Z"]
    xb = range(len(axes_lbl))
    for j, topo in enumerate(topos):
        off = (j - (n - 1) / 2) * width
        ha = halo_axis(runs[topo])
        xs = [xi + off for xi in xb]
        bars = axB.bar(xs, [ha[a] for a in axes_lbl], width=width,
                       color=TOPO_COLOR.get(topo, f"C{j}"), alpha=0.9,
                       label=TOPO_LABEL.get(topo, topo))
        for rect, a in zip(bars, axes_lbl):
            axB.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                     f"{ha[a]:.1f}", ha="center", va="bottom", fontsize=7)
    axB.set_xticks(list(xb))
    axB.set_xticklabels([f"halo {a}" for a in axes_lbl])
    axB.set_ylabel("Tempo acumulado, 40 passos (s)")
    axB.set_title("Halo por eixo — corte inter-nó em negrito")
    axB.grid(True, axis="y", ls=":", alpha=0.4)
    axB.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, out)


def _save(fig, out):
    fig.savefig(out)
    png = out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"saved: {out} and {png}")


def print_summary(runs):
    for topo in ordered_topos(runs):
        rl = runs[topo]
        step = statistics.fmean([s for _, s in rl])
        halo = avg(rl, "halo_total")
        hx, hy = avg(rl, "halo_X"), avg(rl, "halo_Y")
        hz = avg(rl, "halo_Z_phase1") + avg(rl, "halo_Z_phase2")
        print(f"{topo:>10}  runs={len(rl)}  step_total={step:7.2f}s  "
              f"halo_total={halo:6.2f}s ({halo/step*100:4.1f}%)  "
              f"[X={hx:.2f} Y={hy:.2f} Z={hz:.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir com os prof_*/ copiados do cluster")
    ap.add_argument("--grid", default="3200x1280x1280", help="filtra por grid global (ou '' p/ todas)")
    ap.add_argument("--nodes", type=int, default=8, help="filtra por nº de nós (0 p/ todos)")
    ap.add_argument("--out-dir", default=None, help="onde salvar as figuras (default: --root)")
    args = ap.parse_args()

    runs = collect(args.root, args.grid or None, args.nodes or None)
    if not runs:
        raise SystemExit(f"nenhum prof_* casou (root={args.root}, grid={args.grid}, nodes={args.nodes})")

    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)
    print_summary(runs)
    plot_time(runs, os.path.join(out_dir, "topo8_time.pdf"))
    plot_profile(runs, os.path.join(out_dir, "topo8_profile.pdf"))


if __name__ == "__main__":
    main()
