#!/bin/bash
# Lança a matriz de configs nccl-tests em 1, 2 e 4 nós — alinhada com o
# experimento de profiling (que rodou em 1n/2n/4n).
#
# Configs:
#   - nvlink_1n4g  → α,β puros do NVLink (ring intra-nó, 4 GPUs, sem IB)
#   - ib_2n1g      → α,β puros do IB     (1 GPU/nó força tráfego só pela IB)
#   - mixed_2n4g   → bandwidth efetivo no regime do profile 2n (4 GPUs/nó)
#   - mixed_4n4g   → bandwidth efetivo no regime do profile 4n (4 GPUs/nó)
#
# Uso: nohup bash scripts/run_nccl_tests_all.sh > nccl_wrapper.log 2>&1 & disown

set -e
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPTS_DIR}/run_nccl_tests.sbatch"

declare -a CONFIGS=(
    # label          nodes  gpus_per_node
    "nvlink_1n4g     1      4"
    "ib_2n1g         2      1"
    "mixed_2n4g      2      4"
    "mixed_4n4g      4      4"
)

for cfg in "${CONFIGS[@]}"; do
    read -r LABEL NODES GPN <<< "$cfg"
    echo "=== launching $LABEL  (nodes=$NODES, gpus/node=$GPN) ==="
    sbatch --wait \
        --nodes=$NODES \
        --ntasks-per-node=$GPN \
        "$SBATCH" "$LABEL"
    echo "=== done  $LABEL ==="
done

echo "All nccl-tests configs launched."
ls -1d /scratch/g-assimila/rodrigo.machado2/logs_comm/nccl_runs/*/
