#!/bin/bash
# Exp 0 — Validação isend/irecv (PAPER_CONTEXT, Seção 0).
# Uma execução do solver (1 estratégia × 1 run × 2 nós).
#
# Args posicionais:
#   $1 NX  $2 NY  $3 NZ  $4 RUN_ID  $5 CSV_FILE  $6 TOPOLOGY  $7 SAVE_FLAG
#   $8 HALO_STRATEGY  $9 STEPS
#
# Para fila com MaxJobs > 1 (sequana_gpu) ou jobs sequenciais (sequana_gpu_dev),
# cada run usa um WORK_DIR próprio em $SCRATCH/logs_comm/exp0_runs/, evitando
# race em FPS/.
#SBATCH --exclusive
#SBATCH -p sequana_gpu_dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=00:15:00
#SBATCH -J ai4pde_exp0
#SBATCH -o /scratch/g-assimila/rodrigo.machado2/logs_comm/exp0_%j.out
#SBATCH -e /scratch/g-assimila/rodrigo.machado2/logs_comm/exp0_%j.err

set -e
module load python/3.10.16_sequana

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_SOCKET_IFNAME=ib0
export RDZV_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
# Variáveis adicionais sugeridas pelo PAPER_CONTEXT (Seção 10) caso async falhe:
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_P2P_LEVEL=NVL
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

NX=$1
NY=$2
NZ=$3
RUN_ID=$4
CSV_FILE=$5
TOPOLOGY=$6
SAVE_FLAG=$7
HALO_STRATEGY=$8
STEPS=$9
PROFILE=${10:-0}

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
SRC_DIR="${SCRATCH_DIR}/AI4PDEs-MultiGPU-MultiNode/src"
LOG_FILE="${SCRATCH_DIR}/logs_comm/exp0_${SLURM_JOB_ID}.out"

# Cada run em CWD próprio para isolar FPS/ entre execuções concorrentes.
WORK_DIR="${SCRATCH_DIR}/logs_comm/exp0_runs/${HALO_STRATEGY}_r${RUN_ID}_${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# main.py importa halo_exchange/solver/etc. por nome — adicionar src ao path.
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

set +e
srun python3 -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "${SRC_DIR}/main.py" \
    --save $SAVE_FLAG \
    --debug 1 \
    --nx $NX --ny $NY --nz $NZ \
    --topology $TOPOLOGY \
    --halo-strategy $HALO_STRATEGY \
    --steps $STEPS \
    --profile $PROFILE
SRUN_RC=$?
set -e

# Extração de métricas do log do SLURM.
TIME_S=$(grep "Execution_time," "$LOG_FILE" | tail -n 1 | awk -F',' '{print $2}' | tr -d 's ' || echo "ERROR")
SAVE_TIME_S=$(grep "Save_time," "$LOG_FILE" | tail -n 1 | awk -F',' '{print $2}' | tr -d 's ' || echo "0.00")

TOPO_UPPER=${TOPOLOGY^^}

if [ "$SRUN_RC" -eq 0 ] && [ "$TIME_S" != "ERROR" ] && [ -n "$TIME_S" ]; then
    STATUS="SUCESSO"
else
    STATUS="FALHA"
    TIME_S=${TIME_S:-"-"}
    SAVE_TIME_S=${SAVE_TIME_S:-"-"}
fi

echo "${SLURM_JOB_ID},${STATUS},${TOPO_UPPER},${SLURM_JOB_NUM_NODES},${NX},${NY},${NZ},${RUN_ID},${TIME_S},${SAVE_TIME_S},${SAVE_FLAG},${HALO_STRATEGY},${STEPS}" >> "${CSV_FILE}"
