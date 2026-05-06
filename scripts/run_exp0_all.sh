#!/bin/bash
# Wrapper Exp 0 — submete 3 estratégias × 5 runs em 2 nós.
# PAPER_CONTEXT, Seção 0: validação isend/irecv vs batch_isend_irecv vs blocking.
#
# Uso:
#   bash run_exp0_all.sh           # defaults: 1024×768×1024, 20 steps, save=1
#   NX=512 NY=512 NZ=512 STEPS=10 bash run_exp0_all.sh   # override via env

set -e

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
LOGS_DIR="${SCRATCH_DIR}/logs_comm"
CSV_FILE="${LOGS_DIR}/exp0_results.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${LOGS_DIR}/exp0_runs"

# Defaults conforme PAPER_CONTEXT Exp 0 (grid 1024×768×1024, 3D).
NX=${NX:-1024}
NY=${NY:-768}
NZ=${NZ:-1024}
TOPOLOGY=${TOPOLOGY:-3d}
SAVE_FLAG=${SAVE_FLAG:-1}     # 1 = salvar para validação numérica
STEPS=${STEPS:-20}            # Exp 0 não precisa 40 steps; 20 já valida correção
N_RUNS=${N_RUNS:-5}
STRATEGIES=${STRATEGIES:-"blocking async_a async_b"}

# Cabeçalho do CSV (sobrescreve se existir).
echo "JOB_ID,STATUS,TOPOLOGY,NODES,NX,NY,NZ,RUN_ID,TIME_S,SAVE_TIME_S,SAVE_FLAG,HALO_STRATEGY,STEPS" > "${CSV_FILE}"

echo "Exp 0 — Validação halo async"
echo "  Grid:       ${NX}×${NY}×${NZ}"
echo "  Topology:   ${TOPOLOGY}"
echo "  Steps:      ${STEPS}"
echo "  Save:       ${SAVE_FLAG}"
echo "  Runs/strat: ${N_RUNS}"
echo "  Strategies: ${STRATEGIES}"
echo "  CSV:        ${CSV_FILE}"
echo "  Logs:       ${LOGS_DIR}/exp0_*.{out,err}"
echo

for STRATEGY in ${STRATEGIES}; do
    for RUN in $(seq 1 ${N_RUNS}); do
        echo "  -> sbatch ${STRATEGY} run ${RUN}/${N_RUNS}"
        sbatch "${SCRIPT_DIR}/run_exp0.sbatch" \
            "${NX}" "${NY}" "${NZ}" "${RUN}" "${CSV_FILE}" \
            "${TOPOLOGY}" "${SAVE_FLAG}" "${STRATEGY}" "${STEPS}"
    done
done

echo
echo "Submetidos. Acompanhe com:"
echo "  squeue --me"
echo "  tail -f ${CSV_FILE}"
echo
echo "Para validar saídas numéricas (após runs concluídos):"
echo "  python3 tools/compare_halo_outputs.py \\"
echo "    \${SCRATCH}/logs_comm/exp0_runs/blocking_r1_*/FPS \\"
echo "    \${SCRATCH}/logs_comm/exp0_runs/async_a_r1_*/FPS"
