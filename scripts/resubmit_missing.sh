#!/bin/bash
# Re-submit any (topology, nodes, io_mode, run) cells of the Exp 1 matrix
# that don't have a SUCESSO entry in exp1_results.csv. Used after a wave
# finishes to fill in the gaps (timeouts, ECC failures, etc.) without
# re-running everything.
#
# Usage:
#   bash scripts/resubmit_missing.sh wave1
#   bash scripts/resubmit_missing.sh wave2
#   bash scripts/resubmit_missing.sh all
#   bash scripts/resubmit_missing.sh --dry-run all   # print what would submit
#
# Override via env (defaults shown):
#   STEPS=40  PARTITION=sequana_gpu  MAX_INFLIGHT=20  bash scripts/resubmit_missing.sh wave1

set -u

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
LOGS_DIR="${SCRATCH_DIR}/logs_comm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEPS=${STEPS:-40}
PARTITION=${PARTITION:-sequana_gpu}
MAX_INFLIGHT=${MAX_INFLIGHT:-20}

CSV_FILE="${LOGS_DIR}/exp1_results.csv"
MANIFEST="${LOGS_DIR}/exp1_manifest.tsv"

NX=1024
NY=768

if [ ! -f "${CSV_FILE}" ]; then
    echo "ERRO: ${CSV_FILE} não encontrado." >&2
    exit 1
fi

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi
WAVE=${1:-help}

ts() { date +%H:%M:%S; }

wait_for_slot() {
    while : ; do
        local inflight
        inflight=$(squeue --me -h -p "${PARTITION}" -o "%i" 2>/dev/null | wc -l)
        if [ "${inflight}" -lt "${MAX_INFLIGHT}" ]; then
            return 0
        fi
        sleep 30
    done
}

# Returns the list of run_ids in the CSV that have SUCESSO for the given cell.
successful_runs() {
    local topo=$1 nodes=$2 io=$3
    awk -F',' -v topo="${topo}" -v io="${io}" -v nodes="${nodes}" \
        '$2=="SUCESSO" && $3==topo && $5==nodes && $4==io {print $9}' \
        "${CSV_FILE}" | sort -un
}

# Print run_ids that are expected but missing for a given cell.
missing_runs() {
    local topo=$1 nodes=$2 io=$3 n_runs=$4
    local got expected
    got=$(successful_runs "${topo}" "${nodes}" "${io}")
    expected=$(seq 1 "${n_runs}")
    comm -23 <(echo "${expected}" | sort -n) <(echo "${got}" | sort -n)
}

submit_cell() {
    local topo=$1 nodes=$2 io=$3 n_runs=$4
    local nz=$((1024 * nodes))
    local missing
    missing=$(missing_runs "${topo}" "${nodes}" "${io}" "${n_runs}")
    [ -z "${missing}" ] && return 0

    for run in ${missing}; do
        if [ "${DRY_RUN}" -eq 1 ]; then
            echo "[DRY] would submit  ${nodes}n ${topo} ${io} r${run}"
            continue
        fi
        wait_for_slot
        local jobid
        jobid=$(sbatch --parsable --nodes="${nodes}" -p "${PARTITION}" \
            "${SCRIPT_DIR}/run_exp1.sbatch" \
            "${NX}" "${NY}" "${nz}" "${run}" "${topo}" "${io}" "${STEPS}" "${CSV_FILE}")
        local rc=$?
        if [ "${rc}" -ne 0 ] || [ -z "${jobid}" ]; then
            echo "[$(ts)] ERRO ao submeter ${nodes}n ${topo} ${io} r${run} (rc=${rc})"
            continue
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date -Iseconds)" "${jobid}" "resubmit" "${nodes}" "${topo}" "${io}" \
            "${NX}" "${NY}" "${nz}" "${run}" >> "${MANIFEST}"
        echo "[$(ts)] resubmitted ${jobid}  ${nodes}n ${topo} ${io} r${run}"
    done
}

resubmit_wave1() {
    echo "[$(ts)] === Resubmit missing — Wave 1 ==="
    for nodes in 1 2 4 8; do
        for io in none naive async; do
            submit_cell 3d "${nodes}" "${io}" 10
        done
    done
    for nodes in 1 4; do
        for io in none naive async; do
            submit_cell 1d-z "${nodes}" "${io}" 5
        done
    done
}

resubmit_wave2() {
    echo "[$(ts)] === Resubmit missing — Wave 2 ==="
    for nodes in 16 20; do
        for io in none naive async; do
            submit_cell 3d "${nodes}" "${io}" 10
        done
    done
    for nodes in 16; do
        for io in none naive async; do
            submit_cell 1d-z "${nodes}" "${io}" 5
        done
    done
}

case "${WAVE}" in
    wave1)
        echo "Resubmit — WAVE 1 missing cells"
        echo "  CSV:         ${CSV_FILE}"
        echo "  Dry run:     $([ "${DRY_RUN}" -eq 1 ] && echo yes || echo no)"
        echo
        resubmit_wave1
        ;;
    wave2)
        echo "Resubmit — WAVE 2 missing cells"
        echo "  Dry run:     $([ "${DRY_RUN}" -eq 1 ] && echo yes || echo no)"
        echo
        resubmit_wave2
        ;;
    all)
        resubmit_wave1
        resubmit_wave2
        ;;
    help|*)
        echo "Usage: $0 [--dry-run] {wave1|wave2|all}"
        echo
        echo "  Reads ${CSV_FILE}"
        echo "  For each expected (topology, nodes, io_mode, run) cell,"
        echo "  re-submits ONLY runs that don't already have a SUCESSO entry."
        echo
        echo "  --dry-run  prints what would be submitted without doing it"
        exit 1
        ;;
esac

echo
echo "[$(ts)] done."
