#!/bin/bash
# Wrapper for Experiment 1 — Weak scaling with I/O axis.
#
# Two waves:
#   wave1 (priority — small-to-medium node counts, fast turnaround):
#     3D   × {1,2,4,8} nodes × {none,naive,async} × 10 runs = 120 jobs
#     1D-Z × {1,4}    nodes  × {none,naive,async} ×  5 runs =  30 jobs
#   wave2 (large scale, run after wave1):
#     3D   × {16,20} nodes × {none,naive,async} × 10 runs = 60 jobs
#     1D-Z × {16}    nodes × {none,naive,async} ×  5 runs = 15 jobs
#
# Usage:
#   bash scripts/run_exp1_all.sh wave1
#   bash scripts/run_exp1_all.sh wave2
#   bash scripts/run_exp1_all.sh all      # both, in order
#
# Override via env (defaults shown):
#   STEPS=40  PARTITION=sequana_gpu  MAX_INFLIGHT=20  bash scripts/run_exp1_all.sh wave1

set -u

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
LOGS_DIR="${SCRATCH_DIR}/logs_comm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${LOGS_DIR}/exp1_runs"

STEPS=${STEPS:-40}
PARTITION=${PARTITION:-sequana_gpu}
MAX_INFLIGHT=${MAX_INFLIGHT:-20}   # leave headroom under MaxSubmit=24

CSV_FILE="${LOGS_DIR}/exp1_results.csv"
MANIFEST="${LOGS_DIR}/exp1_manifest.tsv"
if [ ! -f "${CSV_FILE}" ]; then
    echo "JOB_ID,STATUS,TOPOLOGY,IO_MODE,NODES,NX,NY,NZ,RUN_ID,STEPS,EXEC_S,SAVE_S,FLUSH_S" > "${CSV_FILE}"
fi
if [ ! -f "${MANIFEST}" ]; then
    printf "submit_ts\tjob_id\twave\tnodes\ttopology\tio_mode\tnx\tny\tnz\trun\n" > "${MANIFEST}"
fi

# Local per GPU fixed at 512x384x1024. NX,NY constant; NZ scales with nodes.
# 3D topology: PX=2, PY=2, PZ=nodes, so NZ_3d = 1024 * nodes.
# 1D-Z keeps the same global NZ as 3D at the same node count, for a fair
# per-problem comparison (local subdomain shape differs between topologies).
NX=1024
NY=768

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

submit_one() {
    local wave=$1 nodes=$2 topo=$3 io=$4 run=$5
    local nz=$((1024 * nodes))
    wait_for_slot

    local jobid
    jobid=$(sbatch --parsable --nodes="${nodes}" -p "${PARTITION}" \
        "${SCRIPT_DIR}/run_exp1.sbatch" \
        "${NX}" "${NY}" "${nz}" "${run}" "${topo}" "${io}" "${STEPS}" "${CSV_FILE}")
    local rc=$?
    if [ "${rc}" -ne 0 ] || [ -z "${jobid}" ]; then
        echo "[$(ts)] ERRO ao submeter ${nodes}n ${topo} ${io} r${run} (rc=${rc})"
        return 1
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -Iseconds)" "${jobid}" "${wave}" "${nodes}" "${topo}" "${io}" \
        "${NX}" "${NY}" "${nz}" "${run}" >> "${MANIFEST}"
    echo "[$(ts)] submitted ${jobid}  ${wave}  ${nodes}n ${topo} ${io} r${run}"
}

submit_wave1() {
    echo "[$(ts)] === Wave 1: small-to-medium scale ==="
    for nodes in 1 2 4 8; do
        for io in none naive async; do
            for run in $(seq 1 10); do
                submit_one wave1 "${nodes}" 3d "${io}" "${run}" || true
            done
        done
    done
    for nodes in 1 4; do
        for io in none naive async; do
            for run in $(seq 1 5); do
                submit_one wave1 "${nodes}" 1d-z "${io}" "${run}" || true
            done
        done
    done
}

submit_wave2() {
    echo "[$(ts)] === Wave 2: large scale ==="
    for nodes in 16 20; do
        for io in none naive async; do
            for run in $(seq 1 10); do
                submit_one wave2 "${nodes}" 3d "${io}" "${run}" || true
            done
        done
    done
    for nodes in 16; do
        for io in none naive async; do
            for run in $(seq 1 5); do
                submit_one wave2 "${nodes}" 1d-z "${io}" "${run}" || true
            done
        done
    done
}

submit_fill_1dz() {
    # Fill the 1D-Z sweep at 2 and 8 nodes so it pairs with the 3D
    # {1,2,4,8,16} node series. Original waves only ran 1D-Z in {1,4,16}.
    echo "[$(ts)] === Fill: 1D-Z at 2 and 8 nodes ==="
    for nodes in 2 8; do
        for io in none naive async; do
            for run in $(seq 1 5); do
                submit_one fill1dz "${nodes}" 1d-z "${io}" "${run}" || true
            done
        done
    done
}

print_summary() {
    echo
    echo "[$(ts)] All requested jobs submitted. Track with:"
    echo "  squeue --me -p ${PARTITION}"
    echo "  tail -f ${CSV_FILE}"
    echo "  wc -l ${MANIFEST}    # total jobs submitted so far"
}

WAVE=${1:-help}

case "${WAVE}" in
    wave1)
        echo "Exp 1 — submitting WAVE 1"
        echo "  Partition:    ${PARTITION}"
        echo "  STEPS:        ${STEPS}"
        echo "  MAX_INFLIGHT: ${MAX_INFLIGHT}"
        echo
        submit_wave1
        print_summary
        ;;
    wave2)
        echo "Exp 1 — submitting WAVE 2"
        echo "  Partition:    ${PARTITION}"
        echo "  STEPS:        ${STEPS}"
        echo "  MAX_INFLIGHT: ${MAX_INFLIGHT}"
        echo
        submit_wave2
        print_summary
        ;;
    all)
        echo "Exp 1 — submitting WAVE 1 then WAVE 2"
        submit_wave1
        submit_wave2
        print_summary
        ;;
    fill1dz)
        echo "Exp 1 — submitting 1D-Z fill (2 and 8 nodes)"
        echo "  Partition:    ${PARTITION}"
        echo "  STEPS:        ${STEPS}"
        echo "  MAX_INFLIGHT: ${MAX_INFLIGHT}"
        echo
        submit_fill_1dz
        print_summary
        ;;
    help|*)
        echo "Usage: $0 {wave1|wave2|all|fill1dz}"
        echo
        echo "  wave1    150 jobs: 3D in {1,2,4,8} nodes + 1D-Z in {1,4}"
        echo "  wave2     75 jobs: 3D in {16,20} nodes  + 1D-Z in {16}"
        echo "  all      submits both, wave1 before wave2"
        echo "  fill1dz   30 jobs: 1D-Z in {2,8} nodes (pairs 1D-Z with 3D)"
        exit 1
        ;;
esac
