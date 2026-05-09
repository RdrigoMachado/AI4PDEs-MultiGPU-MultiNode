#!/bin/bash
# Roda nccl-tests (sendrecv_perf + all_reduce_perf) numa config (nodes × gpus/node).
#
# --nodes e --ntasks-per-node devem vir via CLI:
#   sbatch --nodes=N --ntasks-per-node=G run_nccl_tests.sbatch <LABEL>
#
# Cada task = 1 GPU (-g 1 no nccl-tests). LABEL identifica a config no nome
# do diretório de saída (ex: "nvlink_1n4g", "ib_2n1g", "mixed_4n4g").
#
#SBATCH -p sequana_gpu_dev
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:15:00
#SBATCH -J nccl_tests
#SBATCH -o /scratch/g-assimila/rodrigo.machado2/logs_comm/nccl_%j.out
#SBATCH -e /scratch/g-assimila/rodrigo.machado2/logs_comm/nccl_%j.err

set -e
# Mesmos módulos do build (build_nccl_tests.sh)
module load python/3.10.16_sequana
module load cuda/12.6_sequana
module load openmpi/gnu/4.1.8+cuda-12.6_sequana

LABEL=${1:-unlabeled}
SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
NCCL_TESTS_DIR="${SCRATCH_DIR}/nccl-tests"
OUT_DIR="${SCRATCH_DIR}/logs_comm/nccl_runs/${LABEL}_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

# NCCL via pip wheel — precisa estar no LD_LIBRARY_PATH em runtime
NCCL_HOME=$(python3 -c "import nvidia.nccl as n; print(n.__path__[0])")
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH}"

export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NTASKS=$((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE))

# Captura ambiente para o registro
{
    echo "=== run_nccl_tests metadata ==="
    echo "label=$LABEL"
    echo "jobid=$SLURM_JOB_ID"
    echo "nodes=$SLURM_JOB_NUM_NODES"
    echo "tasks_per_node=$SLURM_NTASKS_PER_NODE"
    echo "total_tasks=$NTASKS"
    echo "nodelist=$SLURM_NODELIST"
    echo "date=$(date -Iseconds)"
} > meta.txt

# sendrecv_perf — para α/β do canal P2P (caso de halo do solver)
echo "=== sendrecv_perf ===" | tee sendrecv.txt
mpirun -np $NTASKS --map-by ppr:$SLURM_NTASKS_PER_NODE:node --bind-to none \
    "${NCCL_TESTS_DIR}/build/sendrecv_perf" \
    -b 1K -e 256M -f 2 -g 1 -w 5 -n 20 -c 0 \
    2>&1 | tee -a sendrecv.txt

# all_reduce_perf — sanity / referência adicional para o modelo
echo "=== all_reduce_perf ===" | tee allreduce.txt
mpirun -np $NTASKS --map-by ppr:$SLURM_NTASKS_PER_NODE:node --bind-to none \
    "${NCCL_TESTS_DIR}/build/all_reduce_perf" \
    -b 1K -e 256M -f 2 -g 1 -w 5 -n 20 -c 0 \
    2>&1 | tee -a allreduce.txt

echo "Done. Output: $OUT_DIR"
