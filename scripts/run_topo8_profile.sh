#!/bin/bash
# Comparação de topologia @ 8 nós, GRID GLOBAL FIXA (mesma grid p/ os dois algos).
#   3d        -> corta Z entre nós (process grid 2x2x8), local 1600x640x160, nlevel 6
#   slab-y-2d -> corta Y entre nós (process grid 2x16x1), local 1600x80x1280, nlevel 5
# Grid 3200x1280x1280 (2.5:1:1, ~164M céls/GPU, pico ~22 GB medido). O nlevel
# DIFERE de propósito: mesma grid, cada decomposição fatia do seu jeito (fato
# reportado, não corrigido). io=none, blocking, profiling ON (via run_profile.sbatch).
#
# Uso (na login do SDumont, dentro da pasta do repo no scratch):
#   bash scripts/run_topo8_profile.sh
#
# Override via env (defaults):
#   RUNS=5 NODES=8 STEPS=40 PARTITION=sequana_gpu WALLTIME=00:15:00
#   GRID="3200 1280 1280"  TOPOS="3d slab-y-2d"
#   EXCLUDE=sdumont8009,sdumont8086,sdumont8008,sdumont8037,sdumont8095
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
GRID=${GRID:-"3200 1280 1280"}
# ECC uncorrectable / lentos conhecidos (sdumont8009 e sdumont8082 confirmados
# ECC em 2026-07-08).
EXCLUDE=${EXCLUDE:-sdumont8009,sdumont8082,sdumont8086,sdumont8008,sdumont8037,sdumont8095}
# shellcheck disable=SC2206
TOPOS=(${TOPOS:-3d slab-y-2d})

read -r NX NY NZ <<< "${GRID}"

MANIFEST="${LOGS_DIR}/topo8_manifest.tsv"
[ -f "${MANIFEST}" ] || printf "submit_ts\tjob_id\ttopology\tnodes\tnx\tny\tnz\trun\n" > "${MANIFEST}"

echo "=== Topo8 profile: grid ${NX}x${NY}x${NZ}, ${NODES} nós, ${RUNS} runs x {${TOPOS[*]}}, profiling ON ==="
echo "    partition=${PARTITION}  walltime=${WALLTIME}  steps=${STEPS}"
echo "    exclude=${EXCLUDE}"
echo "    total jobs: $(( ${#TOPOS[@]} * RUNS ))"

for topo in "${TOPOS[@]}"; do
    for run in $(seq 1 "${RUNS}"); do
        jobid=$(sbatch --parsable --nodes="${NODES}" -p "${PARTITION}" -t "${WALLTIME}" \
            --exclude="${EXCLUDE}" \
            "${SCRIPT_DIR}/run_profile.sbatch" "${NX}" "${NY}" "${NZ}" "${run}" "${topo}" "${STEPS}")
        rc=$?
        if [ "${rc}" -ne 0 ] || [ -z "${jobid}" ]; then
            echo "    ERRO submit ${topo} r${run} (rc=${rc})"
            continue
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Iseconds)" "${jobid}" "${topo}" "${NODES}" "${NX}" "${NY}" "${NZ}" "${run}" \
            >> "${MANIFEST}"
        echo "    submitted ${jobid}  ${topo}  r${run}"
    done
done

echo
echo "Acompanhe:   squeue --me -p ${PARTITION}"
echo "Manifesto:   ${MANIFEST}"
echo "Profiles em: ${LOGS_DIR}/profile_runs/prof_<topo>_${NX}x${NY}x${NZ}_${NODES}n_r<run>_<jobid>/"
echo
echo "Quando terminarem, copie os CSVs p/ sua máquina e gere os gráficos:"
echo "  rsync -avz --prune-empty-dirs --include='*/' \\"
echo "    --include='profile_rank*.csv' --include='metrics_rank*.csv' --exclude='*' \\"
echo "    <host>:${LOGS_DIR}/profile_runs/ ~/inpe/profiles/"
echo "  python3 tools/topo8_profile_plots.py --root ~/inpe/profiles"
