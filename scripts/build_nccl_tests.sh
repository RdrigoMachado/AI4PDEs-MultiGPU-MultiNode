#!/bin/bash
# Compila nccl-tests no SCRATCH. Roda no head node (não precisa de sbatch).
#
# Estratégia de versões:
#   - CUDA: módulo cuda/12.6_sequana (forward-compat para compilar; PyTorch usa 12.8 em runtime).
#   - OpenMPI: módulo openmpi/gnu/4.1.8+cuda-12.6_sequana (CUDA-aware).
#   - NCCL: instalado via pip (nvidia-nccl-cu12==2.27.5) — MESMA versão que o
#           solver roda via PyTorch. Não usa o módulo nccl/2.13 (antigo demais,
#           daria microbench não-representativo).
#
# Uso:
#   bash scripts/build_nccl_tests.sh

set -e

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
NCCL_TESTS_DIR="${SCRATCH_DIR}/nccl-tests"
NCCL_PIP_VERSION="2.27.5"

# --- Módulos ---------------------------------------------------------------
module purge
module load python/3.10.16_sequana
module load cuda/12.6_sequana
module load openmpi/gnu/4.1.8+cuda-12.6_sequana
module list

# --- NCCL via pip (mesma versão do PyTorch do solver) ---------------------
# Instala em --user para não mexer no env compartilhado.
pip install --user "nvidia-nccl-cu12==${NCCL_PIP_VERSION}"

NCCL_HOME=$(python3 -c "
import nvidia.nccl as n
print(n.__path__[0])
")
if [ ! -f "$NCCL_HOME/include/nccl.h" ]; then
    echo "ERRO: nccl.h não encontrado em $NCCL_HOME/include/" >&2
    ls -la "$NCCL_HOME/include" || true
    exit 1
fi

# Wheel nvidia-nccl-cu12 traz só libnccl.so.2 (sem libnccl.so).
# Makefile do nccl-tests linka com -lnccl, que exige libnccl.so.
NCCL_REAL_LIB=$(ls "$NCCL_HOME"/lib/libnccl.so.* 2>/dev/null | head -1)
if [ -z "$NCCL_REAL_LIB" ]; then
    echo "ERRO: libnccl.so.* não encontrada em $NCCL_HOME/lib/" >&2
    ls -la "$NCCL_HOME/lib" || true
    exit 1
fi
NCCL_REAL_BASENAME=$(basename "$NCCL_REAL_LIB")
[ -e "$NCCL_HOME/lib/libnccl.so" ] || ln -s "$NCCL_REAL_BASENAME" "$NCCL_HOME/lib/libnccl.so"
echo "NCCL lib: $NCCL_REAL_BASENAME (+ symlink libnccl.so)"

# --- CUDA / MPI homes ------------------------------------------------------
CUDA_HOME=${CUDA_HOME:-${CUDA_DIR:-}}
if [ -z "$CUDA_HOME" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME=$(dirname $(dirname $(which nvcc)))
fi
MPI_HOME=${MPI_HOME:-$(dirname $(dirname $(which mpicc)))}

echo "CUDA_HOME=$CUDA_HOME"
echo "NCCL_HOME=$NCCL_HOME"
echo "MPI_HOME=$MPI_HOME"

# --- Clone -----------------------------------------------------------------
if [ ! -d "$NCCL_TESTS_DIR" ]; then
    git clone https://github.com/NVIDIA/nccl-tests "$NCCL_TESTS_DIR"
fi
cd "$NCCL_TESTS_DIR"
git pull || true

# --- Build -----------------------------------------------------------------
make clean || true
make MPI=1 \
    CUDA_HOME="$CUDA_HOME" \
    NCCL_HOME="$NCCL_HOME" \
    MPI_HOME="$MPI_HOME" \
    -j 8

echo ""
echo "Build OK. Binários em: $NCCL_TESTS_DIR/build/"
ls -1 "$NCCL_TESTS_DIR/build/"

# --- Sanity rápido (só single-GPU; multi-GPU/multi-node via sbatch) -------
echo ""
echo "Sanity check (1 GPU):"
"${NCCL_TESTS_DIR}/build/sendrecv_perf" -b 1K -e 1M -f 4 -g 1 -n 5 -w 2 -c 0 || \
    echo "(sanity falhou — comum se não houver GPU no head node; rode via sbatch)"
