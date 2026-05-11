#!/bin/bash
# Sanity check for --io-mode async: 1 node, small grid, fast turnaround.
# Validates that every rank's ThreadPoolExecutor writes complete files
# (uniform sizes per field) and that Save_time / Flush_time are reported.
# Expected on 1 node × 4 GPUs × 4 save steps: 16 files per field.
#
# Submit:  sbatch scripts/smoke_io_async.sbatch
#
#SBATCH -p sequana_gpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=00:10:00
#SBATCH -J ai4pde_smoke_io
#SBATCH -o /scratch/g-assimila/rodrigo.machado2/logs_comm/smoke_io_%j.out
#SBATCH -e /scratch/g-assimila/rodrigo.machado2/logs_comm/smoke_io_%j.err

set -e
module load python/3.10.16_sequana

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_SOCKET_IFNAME=ib0
export RDZV_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
SRC_DIR="${SCRATCH_DIR}/AI4PDEs-MultiGPU-MultiNode/src"

WORK_DIR="${SCRATCH_DIR}/logs_comm/smoke_runs/io_async_${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

NX=512
NY=256
NZ=128

srun python3 -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "${SRC_DIR}/main.py" \
    --io-mode async \
    --debug 1 \
    --nx $NX --ny $NY --nz $NZ \
    --topology 1d-z \
    --halo-strategy blocking \
    --steps 40

echo
echo "=== FPS/ contents ==="
ls -la FPS/ 2>/dev/null | head -40
echo
echo "=== file count by prefix ==="
for prefix in u v w p; do
    count=$(ls FPS/${prefix}*.npy 2>/dev/null | wc -l)
    echo "  ${prefix}: ${count} files"
done
echo
echo "=== sanity check: uniform sizes per field ==="
for prefix in u v w p; do
    sizes=$(stat -c%s FPS/${prefix}*.npy 2>/dev/null | sort -u)
    echo "  ${prefix}: distinct sizes = ${sizes}"
done
