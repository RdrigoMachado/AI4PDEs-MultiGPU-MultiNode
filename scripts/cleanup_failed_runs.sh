#!/bin/bash
# Limpeza pós-Exp 1: mantém apenas dirs de profile cujo JOB_ID está marcado
# como SUCESSO em exp1_results.csv, remove o resto (FALHA + órfãos sem CSV)
# e reescreve o CSV sem as linhas FALHA.
#
# Por quê: tools/aggregate_profile.py contabiliza qualquer dir com
# profile_rank*.csv presente; jobs FALHA que crasharam após algum rank ter
# feito dump deixam profile parcial e inflam n_runs. Cruzar com o CSV
# (status=SUCESSO) é o filtro mais robusto e não depende de RUN_ID.
#
# Uso (no SDumont):
#   bash scripts/cleanup_failed_runs.sh             # dry-run (default)
#   bash scripts/cleanup_failed_runs.sh --apply     # executa de fato
#
# Variáveis:
#   LOGS_DIR — pasta com exp1_results.csv e exp1_runs/
#              (default: /scratch/g-assimila/rodrigo.machado2/logs_comm)

set -u

LOGS_DIR="${LOGS_DIR:-/scratch/g-assimila/rodrigo.machado2/logs_comm}"
CSV_FILE="${LOGS_DIR}/exp1_results.csv"
RUNS_DIR="${LOGS_DIR}/exp1_runs"

APPLY=0
case "${1:-}" in
    "")        APPLY=0 ;;
    --apply)   APPLY=1 ;;
    -h|--help)
        sed -n '2,18p' "$0" | sed 's/^# \?//'
        exit 0 ;;
    *)
        echo "ERRO: argumento desconhecido: $1" >&2
        echo "Uso: $0 [--apply]" >&2
        exit 1 ;;
esac

[ -f "${CSV_FILE}" ] || { echo "ERRO: não achei ${CSV_FILE}" >&2; exit 1; }
[ -d "${RUNS_DIR}" ] || { echo "ERRO: não achei ${RUNS_DIR}" >&2; exit 1; }

# Set de JOB_IDs SUCESSO (lookup O(1) via assoc array)
declare -A SUCC
while read -r id; do
    [ -n "${id}" ] && SUCC["${id}"]=1
done < <(awk -F',' 'NR>1 && $2=="SUCESSO" {print $1}' "${CSV_FILE}")

n_succ=${#SUCC[@]}

# Classifica dirs em exp1_runs/. JOB_ID = último campo do nome (após o último '_').
TO_REMOVE=()
kept=0
for d in "${RUNS_DIR}"/exp1_*; do
    [ -d "${d}" ] || continue
    base=$(basename "${d}")
    jid=${base##*_}
    if [ -n "${SUCC[${jid}]:-}" ]; then
        kept=$((kept + 1))
    else
        TO_REMOVE+=("${d}")
    fi
done

falha_lines=$(awk -F',' 'NR>1 && $2=="FALHA"' "${CSV_FILE}")
falha_count=$(printf '%s\n' "${falha_lines}" | grep -c . || true)

echo "=== Resumo ==="
echo "  CSV:               ${CSV_FILE}"
echo "  Runs dir:          ${RUNS_DIR}"
echo "  JOB_IDs SUCESSO:   ${n_succ}"
echo "  Dirs mantidos:     ${kept}"
echo "  Dirs a remover:    ${#TO_REMOVE[@]}"
echo "  Linhas FALHA CSV:  ${falha_count}"
echo

if [ "${#TO_REMOVE[@]}" -gt 0 ]; then
    echo "=== Dirs que serão removidos (até 20 primeiros) ==="
    printf '  %s\n' "${TO_REMOVE[@]}" | head -n 20
    if [ "${#TO_REMOVE[@]}" -gt 20 ]; then
        echo "  ... (+$((${#TO_REMOVE[@]} - 20)) mais)"
    fi
    echo
fi

if [ "${falha_count}" -gt 0 ]; then
    echo "=== Linhas FALHA que serão removidas do CSV (até 10) ==="
    printf '%s\n' "${falha_lines}" | head -n 10 | sed 's/^/  /'
    if [ "${falha_count}" -gt 10 ]; then
        echo "  ... (+$((falha_count - 10)) mais)"
    fi
    echo
fi

if [ "${APPLY}" -eq 0 ]; then
    echo "DRY-RUN — nada foi alterado."
    echo "Para aplicar:  $0 --apply"
    exit 0
fi

echo "=== Aplicando ==="

ts=$(date +%Y%m%d_%H%M%S)
backup="${CSV_FILE}.bak.${ts}"
cp "${CSV_FILE}" "${backup}"
echo "  backup do CSV: ${backup}"

tmpcsv=$(mktemp)
awk -F',' 'NR==1 || $2=="SUCESSO"' "${CSV_FILE}" > "${tmpcsv}"
mv "${tmpcsv}" "${CSV_FILE}"
echo "  CSV reescrito (sem FALHA)."

removed=0
for d in "${TO_REMOVE[@]}"; do
    rm -rf "${d}" && removed=$((removed + 1))
done
echo "  ${removed} dir(s) removido(s)."

echo
echo "Pronto. Reagregue com:"
echo "  python3 tools/aggregate_profile.py ${RUNS_DIR}/"
