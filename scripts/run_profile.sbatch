#!/bin/bash
# Profile — uma execução do solver com profile=1, save=0, blocking.
# PAPER_CONTEXT Seções 0/3: caracterização de gargalo via tempo + memória.
#
# --nodes deve vir via CLI: sbatch --nodes=N run_profile.sbatch ARGS
# (não há diretiva #SBATCH --nodes para permitir variação pela matriz).
#
# Args posicionais:
#   $1 NX  $2 NY  $3 NZ  $4 RUN_ID  $5 TOPOLOGY  $6 STEPS
#
#SBATCH --exclusive
#SBATCH -p sequana_gpu_dev
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=00:15:00
#SBATCH -J ai4pde_profile
#SBATCH -o /scratch/g-assimila/rodrigo.machado2/logs_comm/profile_%j.out
#SBATCH -e /scratch/g-assimila/rodrigo.machado2/logs_comm/profile_%j.err

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
STEPS=$6

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
TOPO_DIR="${SCRATCH_DIR}/AI4PDEs-MultiGPU-MultiNode/topology"
NODES=$SLURM_JOB_NUM_NODES

WORK_DIR="${SCRATCH_DIR}/logs_comm/profile_runs/prof_${TOPOLOGY}_${NX}x${NY}x${NZ}_${NODES}n_r${RUN_ID}_${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

export PYTHONPATH="${TOPO_DIR}:${PYTHONPATH}"

srun python3 -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "${TOPO_DIR}/main.py" \
    --save 0 \
    --debug 1 \
    --nx $NX --ny $NY --nz $NZ \
    --topology $TOPOLOGY \
    --halo-strategy blocking \
    --steps $STEPS \
    --profile 1
