#!/bin/bash
# Atribuição de culpa por nó no Exp 1: cruza os jobs SUCESSO/FALHA do CSV com
# os nós em que rodaram e tenta separar "hardware ruim" de "falha sistêmica".
#
# Por que isto não é só "contar FALHA por nó":
#   Quando UM rank morre, o torchrun derruba o job INTEIRO — todos os nós
#   participantes entram na conta de FALHA, não só o culpado. Em jobs de 16/20
#   nós isso contamina dezenas de nós sadios. Para não acusar inocente, o
#   script usa três sinais, do mais forte para o mais fraco:
#
#     1. Root Cause (first observed failure) + exitcode 1  -> culpado real
#        (mesma lógica de find_ecc_nodes.sh; peers em cascata viram SIGTERM).
#     2. Assinatura de hardware no .err do culpado: ECC / Xid / NODE_FAIL.
#     3. Estatística de co-ocorrência: nó que só aparece em FALHA e NUNCA em
#        SUCESSO é suspeito; nó com >=1 SUCESSO é exonerado (a máquina rodou).
#
#   Falhas de TIMEOUT / OOM / NCCL / LAUNCH NÃO incriminam o nó — são
#   sistêmicas (walltime, escala, rendezvous). O script as separa e só põe
#   no --exclude= os nós com assinatura de hardware.
#
# Uso (no SDumont):
#   bash scripts/blame_nodes.sh
#   bash scripts/blame_nodes.sh > /tmp/blame.txt 2>&1

set -u

LOGS_DIR="${LOGS_DIR:-/scratch/g-assimila/rodrigo.machado2/logs_comm}"
CSV_FILE="${LOGS_DIR}/exp1_results.csv"

cd "${LOGS_DIR}" || { echo "ERRO: não achei ${LOGS_DIR}" >&2; exit 1; }
[ -f "${CSV_FILE}" ] || { echo "ERRO: não achei ${CSV_FILE}" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# --- Conjunto de nós de um job -------------------------------------------
# Preferência: sacct (autoritativo). Fallback: hostnames sdumontNNNN no log.
nodes_of_job() {
    local j=$1 nl=""
    if have sacct; then
        nl=$(sacct -j "${j}" -X -n -P -o NodeList 2>/dev/null | head -n1)
    fi
    if [ -n "${nl}" ] && [ "${nl}" != "None assigned" ] && have scontrol; then
        scontrol show hostnames "${nl}" 2>/dev/null && return
    fi
    # Fallback: varre .err e .out por hostnames.
    grep -hoE 'sdumont[0-9]+' "exp1_${j}.err" "exp1_${j}.out" 2>/dev/null | sort -u
}

# --- Estado SLURM do job (pega NODE_FAIL / TIMEOUT direto do scheduler) ---
slurm_state() {
    have sacct || { echo ""; return; }
    sacct -j "$1" -X -n -P -o State 2>/dev/null | head -n1
}

# --- Tipo de falha a partir dos logs -------------------------------------
# Ordem importa: hardware primeiro, sistêmico depois.
fail_type() {
    local j=$1
    local f="exp1_${j}.err" o="exp1_${j}.out" st
    st=$(slurm_state "${j}")
    case "${st}" in
        NODE_FAIL*) echo "NODE_FAIL"; return ;;
    esac
    if grep -qE 'cudaErrorECCUncorrectable|uncorrectable ECC|Xid' "${f}" "${o}" 2>/dev/null; then
        echo "ECC"; return
    fi
    if grep -qE 'CUDA out of memory|OutOfMemoryError|CUDA error: out of memory' "${f}" "${o}" 2>/dev/null; then
        echo "OOM"; return
    fi
    case "${st}" in TIMEOUT*) echo "TIMEOUT"; return ;; esac
    if grep -qE 'DUE TO TIME LIMIT|CANCELLED AT .* DUE TO TIME|time limit' "${f}" "${o}" 2>/dev/null; then
        echo "TIMEOUT"; return
    fi
    if grep -qE 'NCCL.*timeout|Watchdog caught|ProcessGroupNCCL|NCCL WARN|Connection (refused|reset|closed)|Socket Timeout|unhandled (system|cuda) error' "${f}" "${o}" 2>/dev/null; then
        echo "NCCL"; return
    fi
    if grep -qE 'torch.distributed.elastic|rendezvous|store.*timeout|Signal 9 \(SIGKILL\)|Address already in use|failed to connect to' "${f}" "${o}" 2>/dev/null; then
        echo "LAUNCH"; return
    fi
    [ -s "${f}" ] || { echo "SEM_ERR"; return; }
    echo "OUTRO"
}

# --- Culpado real: host do 1o "Root Cause" com exitcode 1 ----------------
# (idêntico a find_ecc_nodes.sh — peers derrubados são SIGTERM, não exit 1)
culprit_node() {
    awk '
        /Root Cause \(first observed failure\)/ { in_rc=1; host=""; next }
        in_rc && /^=+$/ { in_rc=0; host=""; next }
        in_rc && /^[[:space:]]*host[[:space:]]*:[[:space:]]*sdumont[0-9]+/ {
            match($0,/sdumont[0-9]+/); host=substr($0,RSTART,RLENGTH); next }
        in_rc && host!="" && /^[[:space:]]*exitcode[[:space:]]*:[[:space:]]*1([[:space:]]|\()/ {
            print host; exit }
    ' "exp1_${1}.err" 2>/dev/null
}

OK_TALLY=$(mktemp)      # nó  (1 linha por aparição em SUCESSO)
FAIL_TALLY=$(mktemp)    # nó  (1 linha por aparição em FALHA)
HW_SUSPECT=$(mktemp)    # nó  (assinatura de hardware: culpado ECC/Xid/NODE_FAIL)
TYPE_TALLY=$(mktemp)    # "TIPO Nnós"
trap 'rm -f "${OK_TALLY}" "${FAIL_TALLY}" "${HW_SUSPECT}" "${TYPE_TALLY}"' EXIT

echo "=== Jobs FALHA: tipo e culpado ==="
printf "%-10s %-6s %-9s %-6s %s\n" JOB NÓS TIPO IO CULPADO
while IFS=',' read -r jid status topo io nnodes nx ny nz run steps rest; do
    [ "${jid}" = "JOB_ID" ] && continue
    [ -z "${jid}" ] && continue
    job_nodes=$(nodes_of_job "${jid}")
    if [ "${status}" = "SUCESSO" ]; then
        [ -n "${job_nodes}" ] && echo "${job_nodes}" >> "${OK_TALLY}"
        continue
    fi
    [ "${status}" = "FALHA" ] || continue

    ftype=$(fail_type "${jid}")
    echo "${ftype} ${nnodes}n" >> "${TYPE_TALLY}"
    [ -n "${job_nodes}" ] && echo "${job_nodes}" >> "${FAIL_TALLY}"

    culprit=$(culprit_node "${jid}")
    if [ -n "${culprit}" ] && { [ "${ftype}" = "ECC" ] || [ "${ftype}" = "NODE_FAIL" ]; }; then
        echo "${culprit}" >> "${HW_SUSPECT}"
    fi
    printf "%-10s %-6s %-9s %-6s %s\n" \
        "${jid}" "${nnodes}n" "${ftype}" "${io}" "${culprit:-—}"
done < "${CSV_FILE}"

echo
echo "=== Falhas por TIPO × nº de nós (diagnóstico sistêmico vs hardware) ==="
echo "  ECC/NODE_FAIL = hardware do nó.  TIMEOUT/OOM/NCCL/LAUNCH = sistêmico"
echo "  (walltime/escala/rendezvous) — NÃO é nó defeituoso."
sort "${TYPE_TALLY}" | uniq -c | sort -rn | sed 's/^/  /'

echo
echo "=== Nós: SUCESSO vs FALHA (co-ocorrência bruta — inclui peers em cascata) ==="
printf "%-14s %8s %8s  %s\n" NÓ SUCESSO FALHA VEREDITO
sort -u "${OK_TALLY}" "${FAIL_TALLY}" 2>/dev/null | sort -u | while read -r n; do
    [ -z "${n}" ] && continue
    s=$(grep -cxF "${n}" "${OK_TALLY}" 2>/dev/null); s=${s:-0}
    fl=$(grep -cxF "${n}" "${FAIL_TALLY}" 2>/dev/null); fl=${fl:-0}
    hw=$(grep -cxF "${n}" "${HW_SUSPECT}" 2>/dev/null); hw=${hw:-0}
    if [ "${hw}" -gt 0 ]; then
        v="HARDWARE SUSPEITO (culpado ${hw}x ECC/NODE_FAIL)"
    elif [ "${s}" -gt 0 ]; then
        v="ok (rodou ${s}x com sucesso)"
    elif [ "${fl}" -gt 0 ]; then
        v="só em FALHA — provavelmente peer em cascata; checar"
    else
        v="-"
    fi
    printf "%-14s %8s %8s  %s\n" "${n}" "${s}" "${fl}" "${v}"
done

echo
echo "=== Nós com assinatura de HARDWARE (culpado real ECC/NODE_FAIL) ==="
if [ -s "${HW_SUSPECT}" ]; then
    sort "${HW_SUSPECT}" | uniq -c | sort -rn | sed 's/^/  /'
    echo
    echo "Cole no --exclude= do run_exp1.sbatch (junto dos já existentes):"
    sort -u "${HW_SUSPECT}" | paste -sd, -
else
    echo "  Nenhum. As FALHA não têm assinatura de hardware isolável —"
    echo "  veja a tabela TIPO×nós acima: o gargalo é sistêmico, não nó ruim."
fi
