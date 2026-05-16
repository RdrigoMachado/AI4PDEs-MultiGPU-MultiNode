#!/bin/bash
# Triagem de falhas de hardware (ECC uncorrectable) no Exp 1.
#
# Quando um rank morre por ECC o torchrun derruba o job inteiro, e TODO nó
# participante imprime "srun: error:" e seu próprio bloco no .err. Filtrar
# por isso lista o job inteiro, não o culpado.
#
# O único sinal confiável do nó faltoso é o bloco:
#     Root Cause (first observed failure):
#       host      : sdumontXXXX...
#       exitcode  : 1            <- exceção Python real (ECC), não SIGTERM
# Nós derrubados em cascata aparecem como signal/SIGTERM, não exitcode 1.
#
# Uso (no SDumont):
#   bash scripts/find_ecc_nodes.sh

set -u

LOGS_DIR="${LOGS_DIR:-/scratch/g-assimila/rodrigo.machado2/logs_comm}"
CSV_FILE="${LOGS_DIR}/exp1_results.csv"

cd "${LOGS_DIR}" || { echo "ERRO: não achei ${LOGS_DIR}" >&2; exit 1; }
[ -f "${CSV_FILE}" ] || { echo "ERRO: não achei ${CSV_FILE}" >&2; exit 1; }

# Nó(s) faltoso(s) de um .err: host do bloco "Root Cause (first observed
# failure)" cujo exitcode é 1. Ignora hosts mortos por signal/SIGTERM.
culprit_nodes() {
    awk '
        /Root Cause \(first observed failure\)/ { in_rc = 1; host = ""; next }
        in_rc && /^=+$/                         { in_rc = 0; host = ""; next }
        in_rc && /^[[:space:]]*host[[:space:]]*:[[:space:]]*sdumont[0-9]+/ {
            match($0, /sdumont[0-9]+/); host = substr($0, RSTART, RLENGTH); next
        }
        in_rc && host != "" && /^[[:space:]]*exitcode[[:space:]]*:[[:space:]]*1([[:space:]]|\()/ {
            print host; host = ""
        }
    ' "$1" | sort -u
}

failed_jobs=$(awk -F',' '$2 == "FALHA" { print $1 }' "${CSV_FILE}")

echo "=== Jobs FALHA ==="
tally_file=$(mktemp)
for j in ${failed_jobs}; do
    f="exp1_${j}.err"
    if [ ! -f "$f" ]; then
        echo "job ${j}  (sem .err)"
        continue
    fi
    if ! grep -q cudaErrorECCUncorrectable "$f"; then
        echo "job ${j}  OUTRO  (não é ECC — investigar)"
        continue
    fi
    nodes=$(culprit_nodes "$f")
    if [ -z "${nodes}" ]; then
        echo "job ${j}  ECC    -> <Root Cause sem exitcode 1 — checar manualmente>"
    else
        echo "job ${j}  ECC    -> $(echo ${nodes} | tr '\n' ' ')"
        echo "${nodes}" >> "${tally_file}"
    fi
done

echo
echo "=== Frequência por nó culpado (quantos jobs cada um derrubou) ==="
sort "${tally_file}" | uniq -c | sort -rn

echo
echo "=== Nós ruins (únicos) — cole no --exclude do run_exp1.sbatch ==="
sort -u "${tally_file}" | paste -sd, -
rm -f "${tally_file}"
