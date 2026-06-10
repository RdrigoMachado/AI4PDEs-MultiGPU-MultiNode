#!/bin/bash
# Wrapper for the Strong scaling experiment.
#
# Global grid is FIXED at 256 x 256 x 12288; the per-GPU subdomain shrinks as
# nodes grow. 3D decomposition (PX=PY=2, PZ=nodes) keeps local_nx=local_ny=128
# constant, so the multigrid depth is pinned at nlevel=7 for every node count
# (verify in the .out banner: "Max Multigrid Levels: 7").
#
# (nlevel=8 would need local 256 -> nz>=4096 -> 256x256x4096 = 268M cells/GPU
# at 1 node, which OOMs on the 32 GB V100. nlevel=7 with nz=12288 gives 201M
# cells/GPU at 1 node (~24 GB) -- same per-GPU load as the weak experiments --
# and still reaches 16 nodes; see commit history.)
#
#   3D x {1,2,4,8,16} nodes x io=none x RUNS runs
#
# Walltime per node count (the heaviest run is 1 node, ~201M cells/GPU, ~263s):
#   1,2,4 nodes -> 8 min ; 8,16 nodes -> 5 min.
#
# Usage:
#   bash scripts/run_strong_all.sh smoke   # 1 node, 1 run — validate OOM/time first
#   bash scripts/run_strong_all.sh all     # full sweep, 1..16 nodes
#
# Override via env (defaults shown):
#   STEPS=40  RUNS=10  PARTITION=sequana_gpu  MAX_INFLIGHT=20 \
#     bash scripts/run_strong_all.sh all

set -u

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
LOGS_DIR="${SCRATCH_DIR}/logs_comm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${LOGS_DIR}/strong_runs"

STEPS=${STEPS:-40}
RUNS=${RUNS:-10}
PARTITION=${PARTITION:-sequana_gpu}
MAX_INFLIGHT=${MAX_INFLIGHT:-20}   # leave headroom under MaxSubmit=24

# FIXED global grid (strong scaling). NX=NY=256 -> local 128x128 (pins nlevel=7);
# NZ=12288 -> local_nz=12288/nodes, >=128 up to 16 nodes (768 at 16n).
# 128x128x12288 = 201M cells/GPU at 1 node (~24 GB) -- matches the per-GPU load
# of the weak-scaling experiments (512x384x1024 = 201M), so the GPU is loaded
# consistently across the paper. (256x256x4096 / nlevel=8 = 268M OOMs.)
NX=256
NY=256
NZ=12288
TOPO=3d
IO=none

CSV_FILE="${LOGS_DIR}/strong_results.csv"
MANIFEST="${LOGS_DIR}/strong_manifest.tsv"
if [ ! -f "${CSV_FILE}" ]; then
    echo "JOB_ID,STATUS,TOPOLOGY,IO_MODE,NODES,NX,NY,NZ,RUN_ID,STEPS,EXEC_S,SAVE_S,FLUSH_S" > "${CSV_FILE}"
fi
if [ ! -f "${MANIFEST}" ]; then
    printf "submit_ts\tjob_id\tnodes\ttopology\tio_mode\tnx\tny\tnz\trun\twalltime\n" > "${MANIFEST}"
fi

ts() { date +%H:%M:%S; }

# Walltime per node count.
walltime_for() {
    local nodes=$1
    if [ "${nodes}" -le 4 ]; then
        echo "00:08:00"
    else
        echo "00:05:00"
    fi
}

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
    local nodes=$1 run=$2
    local wt
    wt=$(walltime_for "${nodes}")
    wait_for_slot

    local jobid
    jobid=$(sbatch --parsable --nodes="${nodes}" --time="${wt}" -p "${PARTITION}" \
        "${SCRIPT_DIR}/run_strong.sbatch" \
        "${NX}" "${NY}" "${NZ}" "${run}" "${TOPO}" "${IO}" "${STEPS}" "${CSV_FILE}")
    local rc=$?
    if [ "${rc}" -ne 0 ] || [ -z "${jobid}" ]; then
        echo "[$(ts)] ERRO ao submeter ${nodes}n r${run} (rc=${rc})"
        return 1
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -Iseconds)" "${jobid}" "${nodes}" "${TOPO}" "${IO}" \
        "${NX}" "${NY}" "${NZ}" "${run}" "${wt}" >> "${MANIFEST}"
    echo "[$(ts)] submitted ${jobid}  ${nodes}n r${run}  (${wt})"
}

submit_all() {
    echo "[$(ts)] === Strong scaling: ${NX}x${NY}x${NZ}, ${TOPO}, io=${IO} ==="
    for nodes in 1 2 4 8 16; do
        for run in $(seq 1 "${RUNS}"); do
            submit_one "${nodes}" "${run}" || true
        done
    done
}

submit_smoke() {
    echo "[$(ts)] === Smoke: 1 node, 1 run (validate OOM + walltime) ==="
    submit_one 1 1 || true
}

print_summary() {
    echo
    echo "[$(ts)] Submitted. Track with:"
    echo "  squeue --me -p ${PARTITION}"
    echo "  tail -f ${CSV_FILE}"
    echo "  grep -H 'Max Multigrid Levels' ${LOGS_DIR}/strong_*.out   # confirm nlevel=7 everywhere"
}

MODE=${1:-help}

case "${MODE}" in
    smoke)
        echo "Strong scaling — SMOKE (1 node, 1 run)"
        echo "  Grid: ${NX}x${NY}x${NZ}  Topo: ${TOPO}  IO: ${IO}  STEPS: ${STEPS}"
        echo
        submit_smoke
        print_summary
        ;;
    all)
        echo "Strong scaling — FULL sweep (1,2,4,8,16 nodes x ${RUNS} runs)"
        echo "  Grid: ${NX}x${NY}x${NZ}  Topo: ${TOPO}  IO: ${IO}  STEPS: ${STEPS}"
        echo "  Partition:    ${PARTITION}"
        echo "  MAX_INFLIGHT: ${MAX_INFLIGHT}"
        echo
        submit_all
        print_summary
        ;;
    help|*)
        echo "Usage: $0 {smoke|all}"
        echo
        echo "  smoke   1 job:  1 node, 1 run — run this FIRST to confirm it fits in 32GB"
        echo "                  and to measure wall time before the full sweep."
        echo "  all     $((5 * RUNS)) jobs: 3D in {1,2,4,8,16} nodes x ${RUNS} runs, io=none."
        exit 1
        ;;
esac
