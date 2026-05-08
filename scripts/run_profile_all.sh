#!/bin/bash
# Wrapper do experimento de profiling — matriz weak scaling.
# Usa sbatch --wait (compatível com MaxSubmit=1 do dev).
#
# Para cada config, roda 1 warmup (r0, descartado pelo agregador) + N_RUNS
# medidos (r1..rN). Halo é determinística; runs servem só para variância
# de tempo. WORK_DIR isolado por config+run.
#
# Uso (foreground):
#   bash scripts/run_profile_all.sh
#
# Uso (background — desconectar terminal sem matar):
#   nohup bash scripts/run_profile_all.sh > profile_wrapper.log 2>&1 & disown
#   tail -f profile_wrapper.log
#
# Override via env:
#   STEPS=20 N_RUNS=3 TOPOLOGY=3d PARTITION=sequana_gpu_dev bash ...
#   MATRIX="1 1024 1024 1024
#   2 2048 1024 1024" bash ...

set -u

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
LOGS_DIR="${SCRATCH_DIR}/logs_comm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${LOGS_DIR}/profile_runs"

# Matriz weak scaling — volume por GPU constante (~201M cells).
# 1024 cubed deu OOM em V100 32GB (pico real ~30GB / multigrid temps + cudnn
# workspace). 1024x768x1024 fica em ~71% mem; mesma matriz do Exp 0.
# Formato por linha: "NODES NX NY NZ"
MATRIX=${MATRIX:-"
1 1024 768 1024
2 2048 768 1024
4 4096 768 1024
"}
TOPOLOGY=${TOPOLOGY:-3d}
STEPS=${STEPS:-20}
N_RUNS=${N_RUNS:-3}
WARMUP=${WARMUP:-1}
PARTITION=${PARTITION:-sequana_gpu_dev}

ts() { date +%H:%M:%S; }

echo "Profile — caracterização de gargalo (matriz weak)"
echo "  Topology:   ${TOPOLOGY}"
echo "  Steps:      ${STEPS}"
echo "  Warmup r0:  $([ "${WARMUP}" -eq 1 ] && echo sim || echo nao)"
echo "  Runs/cfg:   ${N_RUNS} medidos"
echo "  Partition:  ${PARTITION}"
echo "  Logs:       ${LOGS_DIR}/profile_*.{out,err}"
echo "  Runs dir:   ${LOGS_DIR}/profile_runs/"
echo
echo "Matriz:"
echo "${MATRIX}" | sed 's/^/  /'
echo

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

if [ "${WARMUP}" -eq 1 ]; then
    RUN_RANGE="$(seq 0 ${N_RUNS})"
else
    RUN_RANGE="$(seq 1 ${N_RUNS})"
fi

# Conta total
N_PER_CFG=$(echo "${RUN_RANGE}" | wc -w)
N_CFG=$(echo "${MATRIX}" | grep -cE '^[[:space:]]*[0-9]')
TOTAL=$((N_CFG * N_PER_CFG))

i=0
while read -r NODES NX NY NZ; do
    [ -z "${NODES:-}" ] && continue
    [[ "${NODES}" =~ ^[0-9]+$ ]] || continue

    for RUN in ${RUN_RANGE}; do
        i=$((i + 1))
        TAG="${NODES}n ${NX}x${NY}x${NZ} r${RUN}"
        [ "${RUN}" -eq 0 ] && TAG="${TAG} (warmup)"
        echo "[$(ts)] (${i}/${TOTAL}) sbatch --wait  ${TAG}"

        attempts=0
        while : ; do
            attempts=$((attempts + 1))
            if sbatch --wait --nodes="${NODES}" -p "${PARTITION}" \
                    "${SCRIPT_DIR}/run_profile.sbatch" \
                    "${NX}" "${NY}" "${NZ}" "${RUN}" "${TOPOLOGY}" "${STEPS}"; then
                break
            fi
            rc=$?
            if [ "${attempts}" -ge 5 ]; then
                echo "[$(ts)] ERRO: sbatch falhou ${attempts}× (rc=${rc}); pulando."
                break
            fi
            echo "[$(ts)] sbatch retornou rc=${rc} (tent. ${attempts}); esperando 30s..."
            sleep 30
            wait_for_fila_empty
        done

        echo "[$(ts)] (${i}/${TOTAL}) concluído  ${TAG}"
    done
done <<< "${MATRIX}"

echo
echo "[$(ts)] Tudo concluído."
echo
echo "Para agregar:"
echo "  python3 tools/aggregate_profile.py ${LOGS_DIR}/profile_runs/"
