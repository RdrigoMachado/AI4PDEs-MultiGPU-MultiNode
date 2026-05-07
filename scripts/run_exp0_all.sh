#!/bin/bash
# Wrapper Exp 0 — submete N estratégias × R runs em série.
# Usa `sbatch --wait` para espera ativa (compatível com MaxSubmit=1 do dev).
# PAPER_CONTEXT, Seção 0.
#
# Uso (foreground):
#   bash run_exp0_all.sh
#   NX=512 NY=512 NZ=512 STEPS=10 N_RUNS=3 bash run_exp0_all.sh
#
# Uso (background — desconectar terminal sem matar):
#   nohup bash run_exp0_all.sh > exp0_wrapper.log 2>&1 & disown
#   tail -f exp0_wrapper.log

set -u  # ausentes -> erro; mas NÃO usa -e (queremos continuar mesmo se 1 run falhar)

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
LOGS_DIR="${SCRATCH_DIR}/logs_comm"
CSV_FILE="${LOGS_DIR}/exp0_results.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${LOGS_DIR}/exp0_runs"

NX=${NX:-1024}
NY=${NY:-768}
NZ=${NZ:-1024}
TOPOLOGY=${TOPOLOGY:-3d}
SAVE_FLAG=${SAVE_FLAG:-1}
STEPS=${STEPS:-20}
N_RUNS=${N_RUNS:-5}
STRATEGIES=${STRATEGIES:-"blocking async_a async_b"}
PROFILE=${PROFILE:-0}     # 1 ativa profiler (sync por região; ~0.5% overhead)
PARTITION=${PARTITION:-sequana_gpu_dev}

# CSV: cria header só se vazio/ausente (preserva linhas de runs anteriores).
if [ ! -s "${CSV_FILE}" ]; then
    echo "JOB_ID,STATUS,TOPOLOGY,NODES,NX,NY,NZ,RUN_ID,TIME_S,SAVE_TIME_S,SAVE_FLAG,HALO_STRATEGY,STEPS" > "${CSV_FILE}"
fi

ts() { date +%H:%M:%S; }

echo "Exp 0 — Validação halo async (serializado via sbatch --wait)"
echo "  Grid:       ${NX}×${NY}×${NZ}"
echo "  Topology:   ${TOPOLOGY}"
echo "  Steps:      ${STEPS}"
echo "  Save:       ${SAVE_FLAG}"
echo "  Runs/strat: ${N_RUNS}"
echo "  Strategies: ${STRATEGIES}"
echo "  Profile:    ${PROFILE}"
echo "  Partition:  ${PARTITION}"
echo "  CSV:        ${CSV_FILE}"
echo "  Run logs:   ${LOGS_DIR}/exp0_*.{out,err}"
echo

# Espera qualquer job pré-existente do usuário na partição limpar — evita
# AssocMaxSubmitJobLimit no dev (MaxSubmit=1).
wait_for_fila_empty() {
    local first=1
    while true; do
        local pending
        pending=$(squeue --me -h -p "${PARTITION}" -o "%i" 2>/dev/null | wc -l)
        if [ "${pending}" -eq 0 ]; then
            [ "${first}" -eq 0 ] && echo "[$(ts)] fila ${PARTITION} liberada."
            return 0
        fi
        if [ "${first}" -eq 1 ]; then
            echo "[$(ts)] aguardando ${pending} job(s) prévios em ${PARTITION}..."
            first=0
        fi
        sleep 20
    done
}

wait_for_fila_empty

TOTAL=0
for _s in ${STRATEGIES}; do TOTAL=$((TOTAL + N_RUNS)); done

i=0
for STRATEGY in ${STRATEGIES}; do
    for RUN in $(seq 1 "${N_RUNS}"); do
        i=$((i + 1))
        echo "[$(ts)] (${i}/${TOTAL}) sbatch --wait  strategy=${STRATEGY}  run=${RUN}/${N_RUNS}"

        # Retry de submissão para resistir a transientes (rede slurm, latência).
        attempts=0
        while : ; do
            attempts=$((attempts + 1))
            if sbatch --wait "${SCRIPT_DIR}/run_exp0.sbatch" \
                    "${NX}" "${NY}" "${NZ}" "${RUN}" "${CSV_FILE}" \
                    "${TOPOLOGY}" "${SAVE_FLAG}" "${STRATEGY}" "${STEPS}" "${PROFILE}"; then
                break
            fi
            rc=$?
            if [ "${attempts}" -ge 5 ]; then
                echo "[$(ts)] ERRO: sbatch falhou ${attempts}× (rc=${rc}); pulando run."
                break
            fi
            echo "[$(ts)] sbatch retornou rc=${rc} (tentativa ${attempts}); esperando 30s antes de retentar..."
            sleep 30
            wait_for_fila_empty
        done

        echo "[$(ts)] (${i}/${TOTAL}) concluído  strategy=${STRATEGY}  run=${RUN}"
    done
done

echo
echo "[$(ts)] Tudo concluído. CSV: ${CSV_FILE}"
echo
echo "Validação numérica baseline vs A (após runs):"
echo "  python3 ${SCRATCH_DIR}/AI4PDEs-MultiGPU-MultiNode/tools/compare_halo_outputs.py \\"
echo "    ${LOGS_DIR}/exp0_runs/blocking_r1_*/FPS \\"
echo "    ${LOGS_DIR}/exp0_runs/async_a_r1_*/FPS"
