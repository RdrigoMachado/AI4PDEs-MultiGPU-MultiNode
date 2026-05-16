#!/bin/bash
# Exp 1 — Weak scaling with I/O axis.
# Single solver run for one cell of the matrix.
#
# --nodes is passed via CLI so the same script handles 1..20 nodes.
#
# Positional args:
#   $1 NX  $2 NY  $3 NZ  $4 RUN_ID  $5 TOPOLOGY  $6 IO_MODE  $7 STEPS  $8 CSV_FILE
#
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=00:15:00
# Nós com ECC uncorrectable (hardware). Acrescente novos aqui, separados por vírgula.
#SBATCH --exclude=sdumont8086,sdumont8008
#SBATCH -J ai4pde_exp1
#SBATCH -o /scratch/g-assimila/rodrigo.machado2/logs_comm/exp1_%j.out
#SBATCH -e /scratch/g-assimila/rodrigo.machado2/logs_comm/exp1_%j.err

set -e
module load python/3.10.16_sequana

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_SOCKET_IFNAME=ib0
export RDZV_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

NX=$1
NY=$2
NZ=$3
RUN_ID=$4
TOPOLOGY=$5
IO_MODE=$6
STEPS=$7
CSV_FILE=$8

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
SRC_DIR="${SCRATCH_DIR}/AI4PDEs-MultiGPU-MultiNode/src"
LOG_FILE="${SCRATCH_DIR}/logs_comm/exp1_${SLURM_JOB_ID}.out"

NODES=$SLURM_JOB_NUM_NODES

# Isolate FPS/ and profile CSVs per (config, run, jobid).
WORK_DIR="${SCRATCH_DIR}/logs_comm/exp1_runs/exp1_${TOPOLOGY}_${IO_MODE}_${NX}x${NY}x${NZ}_${NODES}n_r${RUN_ID}_${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

set +e
srun python3 -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "${SRC_DIR}/main.py" \
    --io-mode $IO_MODE \
    --debug 1 \
    --nx $NX --ny $NY --nz $NZ \
    --topology $TOPOLOGY \
    --halo-strategy blocking \
    --steps $STEPS \
    --profile 1
SRUN_RC=$?
set -e

# Pull metrics from the SLURM log.
EXEC_S=$(grep "Execution_time," "$LOG_FILE" | tail -n 1 | awk -F',' '{print $2}' | tr -d 's ' || echo "ERROR")
SAVE_S=$(grep "Save_time,"      "$LOG_FILE" | tail -n 1 | awk -F',' '{print $2}' | tr -d 's ' || echo "0.00")
FLUSH_S=$(grep "Flush_time,"    "$LOG_FILE" | tail -n 1 | awk -F',' '{print $2}' | tr -d 's ' || echo "0.00")

if [ "$SRUN_RC" -eq 0 ] && [ "$EXEC_S" != "ERROR" ] && [ -n "$EXEC_S" ]; then
    STATUS="SUCESSO"
else
    STATUS="FALHA"
    EXEC_S=${EXEC_S:-"-"}
fi

echo "${SLURM_JOB_ID},${STATUS},${TOPOLOGY},${IO_MODE},${NODES},${NX},${NY},${NZ},${RUN_ID},${STEPS},${EXEC_S},${SAVE_S},${FLUSH_S}" >> "${CSV_FILE}"

# Drop the per-rank .npy dumps once metrics are recorded — each rank writes
# ~800 MB per save in the Exp 1 grid, so a single naive 8n run produces
# ~400 GB. The CSV captures the timings; the volumes have no further use.
rm -rf "${WORK_DIR}/FPS"
