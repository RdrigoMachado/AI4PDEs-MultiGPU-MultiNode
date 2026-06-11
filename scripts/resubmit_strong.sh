#!/bin/bash
# Re-submete as células (nodes, run) da matriz do Strong scaling que NÃO têm
# entrada SUCESSO em strong_results.csv. Usado depois de uma onda para tapar
# os buracos (FALHA por ECC/NODE_FAIL/timeout) sem re-rodar o que já deu certo.
#
# Matriz: 3D, io=none, grid FIXO 256x256x12288, nodes {1,2,4,8,16} x RUNS runs.
# (Espelha scripts/run_strong_all.sh — mantenha os dois em sincronia.)
#
# Rode antes scripts/diagnose_strong.sh: se houver nó de hardware suspeito,
# acrescente-o ao --exclude= de run_strong.sbatch ANTES de relançar, senão o
# reenvio pode cair no mesmo nó ruim.
#
# Uso (no SDumont):
#   bash scripts/resubmit_strong.sh --dry-run    # mostra o que faltou
#   bash scripts/resubmit_strong.sh              # submete os faltantes
#
# Override via env (defaults shown):
#   STEPS=40  RUNS=10  PARTITION=sequana_gpu  MAX_INFLIGHT=20 \
#     bash scripts/resubmit_strong.sh

set -u

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
LOGS_DIR="${SCRATCH_DIR}/logs_comm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEPS=${STEPS:-40}
RUNS=${RUNS:-10}
PARTITION=${PARTITION:-sequana_gpu}
MAX_INFLIGHT=${MAX_INFLIGHT:-20}

# Grid FIXO (strong scaling) — idêntico a run_strong_all.sh.
NX=256
NY=256
NZ=12288
TOPO=3d
IO=none

CSV_FILE="${LOGS_DIR}/strong_results.csv"
MANIFEST="${LOGS_DIR}/strong_manifest.tsv"

[ -f "${CSV_FILE}" ] || { echo "ERRO: ${CSV_FILE} não encontrado." >&2; exit 1; }

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

ts() { date +%H:%M:%S; }

walltime_for() {
    local nodes=$1
    if [ "${nodes}" -le 4 ]; then echo "00:08:00"; else echo "00:05:00"; fi
}

wait_for_slot() {
    while : ; do
        local inflight
        inflight=$(squeue --me -h -p "${PARTITION}" -o "%i" 2>/dev/null | wc -l)
        [ "${inflight}" -lt "${MAX_INFLIGHT}" ] && return 0
        sleep 30
    done
}

# run_ids esperados (1..RUNS) sem entrada SUCESSO para um dado nº de nós.
missing_runs() {
    local nodes=$1 n_runs=$2
    awk -F',' -v topo="${TOPO}" -v io="${IO}" -v nodes="${nodes}" -v n="${n_runs}" '
        BEGIN { for (i=1;i<=n;i++) need[i]=1 }
        $2=="SUCESSO" && $3==topo && $5==nodes && $4==io { delete need[$9] }
        END { for (i=1;i<=n;i++) if (i in need) print i }
    ' "${CSV_FILE}"
}

submit_cell() {
    local nodes=$1 n_runs=$2
    local wt missing
    wt=$(walltime_for "${nodes}")
    missing=$(missing_runs "${nodes}" "${n_runs}")
    [ -z "${missing}" ] && { echo "[$(ts)] ${nodes}n: completo (${n_runs}/${n_runs})"; return 0; }

    echo "[$(ts)] ${nodes}n: faltam runs -> ${missing//$'\n'/ }"
    for run in ${missing}; do
        if [ "${DRY_RUN}" -eq 1 ]; then
            echo "  [DRY] submeteria ${nodes}n r${run} (${wt})"
            continue
        fi
        wait_for_slot
        local jobid
        jobid=$(sbatch --parsable --nodes="${nodes}" --time="${wt}" -p "${PARTITION}" \
            "${SCRIPT_DIR}/run_strong.sbatch" \
            "${NX}" "${NY}" "${NZ}" "${run}" "${TOPO}" "${IO}" "${STEPS}" "${CSV_FILE}")
        local rc=$?
        if [ "${rc}" -ne 0 ] || [ -z "${jobid}" ]; then
            echo "  [$(ts)] ERRO ao submeter ${nodes}n r${run} (rc=${rc})"
            continue
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Iseconds)" "${jobid}" "${nodes}" "${TOPO}" "${IO}" \
            "${NX}" "${NY}" "${NZ}" "${run}" "${wt}" >> "${MANIFEST}"
        echo "  [$(ts)] resubmitted ${jobid}  ${nodes}n r${run}  (${wt})"
    done
}

echo "Resubmit — Strong scaling (células faltantes)"
echo "  Grid: ${NX}x${NY}x${NZ}  Topo: ${TOPO}  IO: ${IO}  RUNS: ${RUNS}"
echo "  Dry run: $([ "${DRY_RUN}" -eq 1 ] && echo yes || echo no)"
echo
for nodes in 1 2 4 8 16; do
    submit_cell "${nodes}" "${RUNS}"
done
echo
echo "[$(ts)] done.  Track: squeue --me -p ${PARTITION} ; tail -f ${CSV_FILE}"
