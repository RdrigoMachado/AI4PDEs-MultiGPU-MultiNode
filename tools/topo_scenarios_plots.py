#!/usr/bin/env python3
"""Comparação 3d × slab-y-2d em 3 cenários de forma de grid (@8 nós).

Junta os três regimes num só par de figuras:
  Y=Z base   3200x1280x1280  (faces inter-nó iguais -> só distribuição)
  A: Y menor 1280x320x1280   (slab corta o eixo menor; face ⊥Y >> ⊥Z)
  B: Z maior 1280x1280x3200  (3d corta o eixo maior; face ⊥Y >> ⊥Z)

Saídas:
  scenarios_time.{pdf,png}    exec (min entre runs, robusto ao clock-boost),
                              3d vs slab nos 3 cenários; razão slab/3d anotada.
  scenarios_profile.{pdf,png} (esq.) halo inter-nó — 3d halo_Z vs slab halo_Y;
                              (dir.) halo total como % do passo. O crescimento
                              do halo do slab de A->base->B é o efeito de VOLUME.

Reusa o carregador de topo8_profile_plots.py. Rode depois de copiar os
profile_rank*.csv dos 3 cenários.

Uso:
    python3 tools/topo_scenarios_plots.py \
        --base-root ~/Documentos/inpe/profiling/comparacao \
        --face-root ~/Documentos/inpe/profiling/comparacao/faceshape \
        --out-dir   ~/Documentos/inpe/profiling/comparacao
"""
import argparse
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topo8_profile_plots as tp  # noqa: E402

TOPOS = ["3d", "slab-y-2d"]
COLOR = {"3d": "#4C72B0", "slab-y-2d": "#DD8452"}
LABEL = {"3d": "3D (corta Z)", "slab-y-2d": "slab-y-2d (corta Y)"}


def scenario_stats(root, grid):
    """{topo: {exec_min, exec_mean, halo_ib, halo_total, step, n}} para uma grid."""
    runs = tp.collect(root, grid, 8)
    out = {}
    for topo in tp.ordered_topos(runs):
        rl = runs[topo]
        steps = [r["step"] for r in rl]
        step = tp.avg(rl, "step_total") or st.fmean(steps)
        # halo inter-nó: 3d troca em Z; slab troca em Y.
        halo_ib = (tp.avg(rl, "halo_Z_phase1") + tp.avg(rl, "halo_Z_phase2")
                   if topo == "3d" else tp.avg(rl, "halo_Y"))
        out[topo] = {
            "exec_min": min(steps), "exec_mean": st.fmean(steps),
            "halo_ib": halo_ib, "halo_total": tp.avg(rl, "halo_total"),
            "step": step, "n": len(rl),
        }
    return out


def _grouped_positions(nscen, width):
    import numpy as np
    x = np.arange(nscen)
    return x, {"3d": x - width / 2, "slab-y-2d": x + width / 2}


def plot_time(scen, out):
    labels = [s[0] for s in scen]
    stats = [s[3] for s in scen]
    width = 0.38
    x, pos = _grouped_positions(len(scen), width)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for topo in TOPOS:
        vals = [stt.get(topo, {}).get("exec_min", float("nan")) for stt in stats]
        ax.bar(pos[topo], vals, width=width, color=COLOR[topo], alpha=0.9, label=LABEL[topo])
        for xi, v in zip(pos[topo], vals):
            if v == v:
                ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                            xytext=(0, 2), ha="center", va="bottom", fontsize=8)
    # razão slab/3d por cenário, bem acima dos rótulos de valor
    for xi, stt in zip(x, stats):
        a = stt.get("3d", {}).get("exec_min"); b = stt.get("slab-y-2d", {}).get("exec_min")
        if a and b:
            ax.annotate(f"slab/3d = {b / a:.2f}×", (xi, max(a, b)), textcoords="offset points",
                        xytext=(0, 15), ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color="0.25")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Execution time — min entre runs (s)")
    ax.set_title("Tempo de execução @ 8 nós, 3 formas de grid (mesmo nº de GPUs)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.set_ylim(0, max(v for stt in stats for v in
                       [stt.get(t, {}).get("exec_min", 0) for t in TOPOS]) * 1.28)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    _save(fig, out)


def plot_profile(scen, out):
    labels = [s[0] for s in scen]
    faces = [s[4] for s in scen]     # (face3d, faceslab) em M células, p/ anotar
    stats = [s[3] for s in scen]
    width = 0.38
    x, pos = _grouped_positions(len(scen), width)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- Painel A: halo inter-nó (s) ---
    ib_max = max(stt.get(t, {}).get("halo_ib", 0) for stt in stats for t in TOPOS)
    for topo in TOPOS:
        vals = [stt.get(topo, {}).get("halo_ib", float("nan")) for stt in stats]
        axA.bar(pos[topo], vals, width=width, color=COLOR[topo], alpha=0.9, label=LABEL[topo])
        for xi, v in zip(pos[topo], vals):
            if v == v:
                axA.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                             xytext=(0, 2), ha="center", va="bottom", fontsize=8)
    # razão de área da face inter-nó (slab/3d) vai no rótulo do eixo x
    labels_A = [f"{lab}\nface slab/3d {fs / f3:.1f}×" for lab, (f3, fs) in zip(labels, faces)]
    axA.set_xticks(x); axA.set_xticklabels(labels_A, fontsize=9)
    axA.set_ylim(0, ib_max * 1.25)
    axA.set_ylabel("Halo inter-nó — acumulado 40 passos (s)")
    axA.set_title("Comunicação inter-nó: 3d (halo_Z) vs slab (halo_Y)")
    axA.grid(True, axis="y", ls=":", alpha=0.4)
    axA.legend(frameon=False, loc="upper center", ncol=2)

    # --- Painel B: halo total como % do passo ---
    for topo in TOPOS:
        vals = [100 * stt.get(topo, {}).get("halo_total", 0) / stt[topo]["step"]
                if topo in stt else float("nan") for stt in stats]
        axB.bar(pos[topo], vals, width=width, color=COLOR[topo], alpha=0.9, label=LABEL[topo])
        for xi, v in zip(pos[topo], vals):
            if v == v:
                axB.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=9)
    axB.set_ylabel("Halo total / passo (%)")
    axB.set_title("Peso da comunicação no passo")
    axB.grid(True, axis="y", ls=":", alpha=0.4)
    axB.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    _save(fig, out)


def _save(fig, out):
    fig.savefig(out)
    png = out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"saved: {out} and {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-root", default="/home/rcmachado/Documentos/inpe/profiling/comparacao")
    ap.add_argument("--face-root", default="/home/rcmachado/Documentos/inpe/profiling/comparacao/faceshape")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    # (label, root, grid, stats, (face3d_M, faceslab_M)); faces = área ⊥ do nó em Mcélulas
    defs = [
        ("Y=Z base\n3200×1280×1280", args.base_root, "3200x1280x1280", (3200 * 1280 / 1e6, 3200 * 1280 / 1e6)),
        ("A: Y menor\n1280×320×1280", args.face_root, "1280x320x1280", (1280 * 320 / 1e6, 1280 * 1280 / 1e6)),
        ("B: Z maior\n1280×1280×3200", args.face_root, "1280x1280x3200", (1280 * 1280 / 1e6, 1280 * 3200 / 1e6)),
    ]
    scen = []
    for label, root, grid, faces in defs:
        stats = scenario_stats(root, grid)
        if not stats:
            print(f"AVISO: sem dados p/ {label.strip()} ({grid}) em {root}")
            continue
        scen.append((label, root, grid, stats, faces))
        for topo, s in stats.items():
            print(f"{label.splitlines()[0]:>14} {topo:>10} n={s['n']}  "
                  f"exec_min={s['exec_min']:7.2f}s  halo_ib={s['halo_ib']:6.2f}s  "
                  f"halo%={100 * s['halo_total'] / s['step']:4.1f}")
    if not scen:
        raise SystemExit("nenhum cenário com dados")

    out_dir = args.out_dir or args.base_root
    os.makedirs(out_dir, exist_ok=True)
    plot_time(scen, os.path.join(out_dir, "scenarios_time.pdf"))
    plot_profile(scen, os.path.join(out_dir, "scenarios_profile.pdf"))


if __name__ == "__main__":
    main()
