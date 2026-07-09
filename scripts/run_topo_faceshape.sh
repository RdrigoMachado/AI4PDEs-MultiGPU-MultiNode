#!/bin/bash
# Comparação de topologia @ 8 nós variando a FORMA da face inter-nó (quebra Y=Z).
# Roda os dois algoritmos em duas grids, cada uma alongando um eixo diferente:
#
#   Grid A  1280x320x1280  (Y menor)  -> slab-y-2d corta o eixo MENOR (Y=320)
#           3d   (2x2x8): local 640x160x160,  face ⊥Z do nó = X·Y = 1280x320  = 0.41M
#           slab (2x16x1): local 640x20x1280, face ⊥Y do nó = X·Z = 1280x1280 = 1.64M
#
#   Grid B  1280x1280x3200 (Z maior)  -> 3d corta o eixo MAIOR (Z=3200)
#           3d   (2x2x8): local 640x640x400,  face ⊥Z do nó = X·Y = 1280x1280 = 1.64M
#           slab (2x16x1): local 640x80x3200, face ⊥Y do nó = X·Z = 1280x3200 = 4.10M
#
# Diferente da grid Y=Z (3200x1280x1280), aqui o VOLUME inter-nó difere entre os
# dois algoritmos -> isola o efeito de face/volume, não só de distribuição.
# nlevel NÃO importa (main.py deriva do local; difere entre grids/topos, tudo bem).
# Carga/GPU: Grid A ~16M céls (~2-3 GB, leve), Grid B ~164M (~22 GB).
#
# Uso (na login do SDumont, dentro da pasta do repo no scratch):
#   bash scripts/run_topo_faceshape.sh
#
# Override via env (defaults):
#   RUNS=5 NODES=8 STEPS=40 PARTITION=sequana_gpu WALLTIME=00:15:00
#   TOPOS="3d slab-y-2d"   GRIDS="1280 320 1280 | 1280 1280 3200"  (grids separadas por |)
#   EXCLUDE=sdumont8009,sdumont8082,sdumont8086,sdumont8008,sdumont8037,sdumont8095
set -u

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
LOGS_DIR="${SCRATCH_DIR}/logs_comm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${LOGS_DIR}/profile_runs"

RUNS=${RUNS:-5}
NODES=${NODES:-8}
STEPS=${STEPS:-40}
PARTITION=${PARTITION:-sequana_gpu}
WALLTIME=${WALLTIME:-00:15:00}
# ECC uncorrectable / lentos conhecidos (8009, 8082 confirmados ECC em 2026-07-08).
EXCLUDE=${EXCLUDE:-sdumont8009,sdumont8082,sdumont8086,sdumont8008,sdumont8037,sdumont8095}
# shellcheck disable=SC2206
TOPOS=(${TOPOS:-3d slab-y-2d})

# Grids separadas por "|" (default: A e B). Cada uma é "NX NY NZ".
GRIDS_RAW=${GRIDS:-"1280 320 1280 | 1280 1280 3200"}
IFS='|' read -r -a GRIDS <<< "${GRIDS_RAW}"

MANIFEST="${LOGS_DIR}/faceshape_manifest.tsv"
[ -f "${MANIFEST}" ] || printf "submit_ts\tjob_id\ttopology\tnodes\tnx\tny\tnz\trun\n" > "${MANIFEST}"

echo "=== Faceshape: ${#GRIDS[@]} grids x {${TOPOS[*]}} x ${RUNS} runs @ ${NODES} nós, profiling ON ==="
echo "    partition=${PARTITION}  walltime=${WALLTIME}  steps=${STEPS}"
echo "    exclude=${EXCLUDE}"
echo "    total jobs: $(( ${#GRIDS[@]} * ${#TOPOS[@]} * RUNS ))"

for grid in "${GRIDS[@]}"; do
    read -r NX NY NZ <<< "${grid}"
    for topo in "${TOPOS[@]}"; do
        for run in $(seq 1 "${RUNS}"); do
            jobid=$(sbatch --parsable --nodes="${NODES}" -p "${PARTITION}" -t "${WALLTIME}" \
                --exclude="${EXCLUDE}" \
                "${SCRIPT_DIR}/run_profile.sbatch" "${NX}" "${NY}" "${NZ}" "${run}" "${topo}" "${STEPS}")
            rc=$?
            if [ "${rc}" -ne 0 ] || [ -z "${jobid}" ]; then
                echo "    ERRO submit ${topo} ${NX}x${NY}x${NZ} r${run} (rc=${rc})"
                continue
            fi
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$(date -Iseconds)" "${jobid}" "${topo}" "${NODES}" "${NX}" "${NY}" "${NZ}" "${run}" \
                >> "${MANIFEST}"
            echo "    submitted ${jobid}  ${topo}  ${NX}x${NY}x${NZ}  r${run}"
        done
    done
done

echo
echo "Acompanhe:   squeue --me -p ${PARTITION}"
echo "Manifesto:   ${MANIFEST}"
echo "Profiles em: ${LOGS_DIR}/profile_runs/prof_<topo>_<grid>_${NODES}n_r<run>_<jobid>/"
echo
echo "Depois, copie os CSVs e gere os gráficos POR GRID:"
echo "  python3 tools/topo8_profile_plots.py --root <dir> --grid 1280x320x1280   --out-dir <dir>/gridA"
echo "  python3 tools/topo8_profile_plots.py --root <dir> --grid 1280x1280x3200  --out-dir <dir>/gridB"
