#!/bin/bash
# Triagem de falhas de hardware (ECC uncorrectable) no Exp 1.
#
# Para cada job FALHA no exp1_results.csv, inspeciona o .err:
#   - se for cudaErrorECCUncorrectable -> extrai o nó físico culpado
#   - senão -> marca como "outro" (timeout / bug real, vale investigar)
#
# No fim imprime a lista única de nós ruins, já no formato do --exclude.
#
# Uso (no SDumont):
#   bash scripts/find_ecc_nodes.sh

set -u

LOGS_DIR="${LOGS_DIR:-/scratch/g-assimila/rodrigo.machado2/logs_comm}"
CSV_FILE="${LOGS_DIR}/exp1_results.csv"

cd "${LOGS_DIR}" || { echo "ERRO: não achei ${LOGS_DIR}" >&2; exit 1; }
[ -f "${CSV_FILE}" ] || { echo "ERRO: não achei ${CSV_FILE}" >&2; exit 1; }

failed_jobs=$(awk -F',' '$2 == "FALHA" { print $1 }' "${CSV_FILE}")

echo "=== Jobs FALHA ==="
for j in ${failed_jobs}; do
    f="exp1_${j}.err"
    if [ ! -f "$f" ]; then
        echo "job ${j}  (sem .err)"
        continue
    fi
    if grep -q cudaErrorECCUncorrectable "$f"; then
        hosts=$(grep -oE 'srun: error: sdumont[0-9]+|host *: *sdumont[0-9]+' "$f" \
                | grep -oE 'sdumont[0-9]+' | sort -u | tr '\n' ' ')
        echo "job ${j}  ECC    -> ${hosts:-<sem linha host>}"
    else
        echo "job ${j}  OUTRO  (não é ECC — investigar)"
    fi
done

echo
echo "=== Nós ruins (únicos) — cole no --exclude do run_exp1.sbatch ==="
for j in ${failed_jobs}; do
    f="exp1_${j}.err"
    [ -f "$f" ] || continue
    grep -q cudaErrorECCUncorrectable "$f" || continue
    grep -oE 'srun: error: sdumont[0-9]+|host *: *sdumont[0-9]+' "$f" \
        | grep -oE 'sdumont[0-9]+'
done | sort -u | paste -sd, -
