#!/bin/bash
# Diagnóstico do Strong scaling (strong_results.csv). Responde três perguntas:
#
#   1. O que falhou e por quê? (tipo de falha + nó culpado real)
#   2. A falha foi de um nó específico? (ECC/NODE_FAIL -> hardware)
#   3. A variação de tempo (runs lentos vs rápidos no MESMO nº de nós) é de
#      um nó específico? -> ESTE é o ponto novo deste script.
#
# Por que (3) importa neste experimento:
#   Em strong scaling o grid global é FIXO e os ranks rodam sincronizados —
#   se UM nó é mais lento, ele segura o passo de todos e o JOB INTEIRO fica
#   lento (não só o rank dele). Logo, um nó "ruim de desempenho" aparece como
#   tempo de execução inflado em TODO job que o incluiu. A assinatura é a
#   mesma de blame_nodes.sh, mas o sinal aqui é "job lento" em vez de "job que
#   falhou": um nó que só aparece em runs LENTOS e nunca em runs RÁPIDOS é
#   suspeito de heterogeneidade de hardware.
#
#   Baseline por nº de nós = MENOR EXEC_S observado (o melhor caso de hardware
#   limpo). Um run é "LENTO" se EXEC_S > baseline * (1 + THRESH) (default 5%).
#
# Falha de UM rank derruba o job inteiro via torchrun -> todos os nós entram
# na conta. Por isso o culpado real só é incriminado pelo bloco
# "Root Cause (first observed failure) ... exitcode: 1" (mesma lógica de
# find_ecc_nodes.sh / blame_nodes.sh) e por assinatura de hardware no .err.
#
# Uso (no SDumont):
#   bash scripts/diagnose_strong.sh
#   THRESH=0.05 bash scripts/diagnose_strong.sh > /tmp/diag_strong.txt 2>&1

set -u

LOGS_DIR="${LOGS_DIR:-/scratch/g-assimila/rodrigo.machado2/logs_comm}"
CSV_FILE="${CSV_FILE:-${LOGS_DIR}/strong_results.csv}"
THRESH="${THRESH:-0.05}"   # fração acima do baseline para marcar "LENTO" (5%)

cd "${LOGS_DIR}" || { echo "ERRO: não achei ${LOGS_DIR}" >&2; exit 1; }
[ -f "${CSV_FILE}" ] || { echo "ERRO: não achei ${CSV_FILE}" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# --- Conjunto de nós de um job (com cache) -------------------------------
# Preferência: sacct (autoritativo). Fallback: hostnames no .out/.err.
NODES_CACHE=$(mktemp)
trap 'rm -f "${NODES_CACHE}" "${OK_FAST}" "${OK_SLOW}" "${FAIL_TALLY}" "${HW_SUSPECT}" "${TYPE_TALLY}" "${SLOW_TALLY}" "${FAST_TALLY}" 2>/dev/null' EXIT

nodes_of_job() {
    local j=$1 cached nl=""
    cached=$(grep -m1 "^${j} " "${NODES_CACHE}" 2>/dev/null | cut -d' ' -f2-)
    if [ -n "${cached}" ]; then echo "${cached}"; return; fi
    if have sacct; then
        nl=$(sacct -j "${j}" -X -n -P -o NodeList 2>/dev/null | head -n1)
    fi
    local out=""
    if [ -n "${nl}" ] && [ "${nl}" != "None assigned" ] && have scontrol; then
        out=$(scontrol show hostnames "${nl}" 2>/dev/null | tr '\n' ' ')
    fi
    if [ -z "${out}" ]; then
        out=$(grep -hoE 'sdumont[0-9]+' "strong_${j}.err" "strong_${j}.out" 2>/dev/null | sort -u | tr '\n' ' ')
    fi
    echo "${j} ${out}" >> "${NODES_CACHE}"
    echo "${out}"
}

slurm_state() { have sacct || { echo ""; return; }; sacct -j "$1" -X -n -P -o State 2>/dev/null | head -n1; }

# --- Tipo de falha (hardware primeiro, sistêmico depois) -----------------
fail_type() {
    local j=$1
    local f="strong_${j}.err" o="strong_${j}.out" st
    st=$(slurm_state "${j}")
    case "${st}" in NODE_FAIL*) echo "NODE_FAIL"; return ;; esac
    if grep -qE 'cudaErrorECCUncorrectable|uncorrectable ECC|Xid' "${f}" "${o}" 2>/dev/null; then echo "ECC"; return; fi
    if grep -qE 'CUDA out of memory|OutOfMemoryError|CUDA error: out of memory' "${f}" "${o}" 2>/dev/null; then echo "OOM"; return; fi
    case "${st}" in TIMEOUT*) echo "TIMEOUT"; return ;; esac
    if grep -qE 'DUE TO TIME LIMIT|CANCELLED AT .* DUE TO TIME|time limit' "${f}" "${o}" 2>/dev/null; then echo "TIMEOUT"; return; fi
    if grep -qE 'NCCL.*timeout|Watchdog caught|ProcessGroupNCCL|NCCL WARN|Connection (refused|reset|closed)|Socket Timeout|unhandled (system|cuda) error' "${f}" "${o}" 2>/dev/null; then echo "NCCL"; return; fi
    if grep -qE 'torch.distributed.elastic|rendezvous|store.*timeout|Signal 9 \(SIGKILL\)|Address already in use|failed to connect to' "${f}" "${o}" 2>/dev/null; then echo "LAUNCH"; return; fi
    [ -s "${f}" ] || { echo "SEM_ERR"; return; }
    echo "OUTRO"
}

# --- Culpado real: host do 1o "Root Cause" com exitcode 1 ----------------
culprit_node() {
    awk '
        /Root Cause \(first observed failure\)/ { in_rc=1; host=""; next }
        in_rc && /^=+$/ { in_rc=0; host=""; next }
        in_rc && /^[[:space:]]*host[[:space:]]*:[[:space:]]*sdumont[0-9]+/ {
            match($0,/sdumont[0-9]+/); host=substr($0,RSTART,RLENGTH); next }
        in_rc && host!="" && /^[[:space:]]*exitcode[[:space:]]*:[[:space:]]*1([[:space:]]|\()/ {
            print host; exit }
    ' "strong_${1}.err" 2>/dev/null
}

OK_FAST=$(mktemp); OK_SLOW=$(mktemp)
FAIL_TALLY=$(mktemp); HW_SUSPECT=$(mktemp); TYPE_TALLY=$(mktemp)
SLOW_TALLY=$(mktemp); FAST_TALLY=$(mktemp)

# Baseline (menor EXEC_S) por nº de nós, só de SUCESSO.
declare -A BASE
while IFS=',' read -r jid status topo io nnodes nx ny nz run steps exec_s rest; do
    [ "${jid}" = "JOB_ID" ] && continue
    [ "${status}" = "SUCESSO" ] || continue
    case "${exec_s}" in ''|-|*[!0-9.]*) continue ;; esac
    cur=${BASE[$nnodes]:-}
    if [ -z "${cur}" ] || awk "BEGIN{exit !($exec_s < $cur)}"; then BASE[$nnodes]=$exec_s; fi
done < "${CSV_FILE}"

# =========================================================================
echo "=== 1) Jobs FALHA: tipo e nó culpado ==="
printf "%-10s %-5s %-9s %s\n" JOB NÓS TIPO CULPADO
nfail=0
while IFS=',' read -r jid status topo io nnodes nx ny nz run steps rest; do
    [ "${jid}" = "JOB_ID" ] && continue
    [ -z "${jid}" ] && continue
    job_nodes=$(nodes_of_job "${jid}")
    if [ "${status}" = "FALHA" ]; then
        nfail=$((nfail+1))
        ftype=$(fail_type "${jid}")
        echo "${ftype} ${nnodes}n" >> "${TYPE_TALLY}"
        for n in ${job_nodes}; do echo "${n}" >> "${FAIL_TALLY}"; done
        culprit=$(culprit_node "${jid}")
        if [ -n "${culprit}" ] && { [ "${ftype}" = "ECC" ] || [ "${ftype}" = "NODE_FAIL" ]; }; then
            echo "${culprit}" >> "${HW_SUSPECT}"
        fi
        printf "%-10s %-5s %-9s %s\n" "${jid}" "${nnodes}n" "${ftype}" "${culprit:-—}"
    fi
done < "${CSV_FILE}"
[ "${nfail}" -eq 0 ] && echo "  (nenhuma FALHA no CSV)"

echo
echo "=== 2) Falhas por TIPO × nº de nós (sistêmico vs hardware) ==="
echo "  ECC/NODE_FAIL = hardware do nó.  TIMEOUT/OOM/NCCL/LAUNCH = sistêmico."
if [ -s "${TYPE_TALLY}" ]; then sort "${TYPE_TALLY}" | uniq -c | sort -rn | sed 's/^/  /'; else echo "  —"; fi

# =========================================================================
echo
echo "=== 3) Variação de desempenho por nº de nós (baseline = menor EXEC_S) ==="
echo "  Um run é LENTO se EXEC_S > baseline * (1 + ${THRESH})."
for nodes in $(printf '%s\n' "${!BASE[@]}" | sort -n); do
    base=${BASE[$nodes]}
    lim=$(awk "BEGIN{printf \"%.2f\", $base*(1+$THRESH)}")
    printf "  -- %sn: baseline=%ss  limite_lento=%ss\n" "${nodes}" "${base}" "${lim}"
    while IFS=',' read -r jid status topo io nn nx ny nz run steps exec_s rest; do
        [ "${status}" = "SUCESSO" ] || continue
        [ "${nn}" = "${nodes}" ] || continue
        case "${exec_s}" in ''|-|*[!0-9.]*) continue ;; esac
        tag="rápido"
        list_target="${FAST_TALLY}"; tally_target="${OK_FAST}"
        if awk "BEGIN{exit !($exec_s > $lim)}"; then
            tag="LENTO"; list_target="${SLOW_TALLY}"; tally_target="${OK_SLOW}"
        fi
        job_nodes=$(nodes_of_job "${jid}")
        if [ "${tag}" = "LENTO" ]; then
            printf "       r%-2s %-9s %ss  [%s]  %s\n" "${run}" "${jid}" "${exec_s}" "${tag}" "${job_nodes}"
        fi
        for n in ${job_nodes}; do echo "${n}" >> "${list_target}"; done
        echo "${tag}" >> "${tally_target}"
    done < "${CSV_FILE}"
done
nslow=$(wc -l < "${OK_SLOW}" 2>/dev/null || echo 0)
echo "  (total de runs LENTOS: ${nslow})"

# =========================================================================
echo
echo "=== 4) Atribuição por nó: aparições em runs LENTOS vs RÁPIDOS ==="
echo "  Nó só em LENTO (e nunca em RÁPIDO) => suspeito de hardware mais lento."
printf "  %-14s %6s %6s  %s\n" NÓ LENTO RÁPIDO VEREDITO
SLOW_NODES=$(mktemp)
sort -u "${SLOW_TALLY}" "${FAST_TALLY}" 2>/dev/null | while read -r n; do
    [ -z "${n}" ] && continue
    sl=$(grep -cxF "${n}" "${SLOW_TALLY}" 2>/dev/null); sl=${sl:-0}
    fa=$(grep -cxF "${n}" "${FAST_TALLY}" 2>/dev/null); fa=${fa:-0}
    if [ "${sl}" -gt 0 ] && [ "${fa}" -eq 0 ]; then
        v="SUSPEITO (só em runs lentos)"; echo "${n}" >> "${SLOW_NODES}"
    elif [ "${sl}" -gt 0 ] && [ "${fa}" -gt 0 ]; then
        v="misto (lento ${sl}x, rápido ${fa}x)"
    else
        v="ok"
    fi
    printf "  %-14s %6s %6s  %s\n" "${n}" "${sl}" "${fa}" "${v}"
done

echo
echo "=== 5) Veredito: nós a excluir no --exclude= do run_strong.sbatch ==="
ECC_LIST=""; SLOW_LIST=""
[ -s "${HW_SUSPECT}" ] && ECC_LIST=$(sort -u "${HW_SUSPECT}" | paste -sd, -)
[ -s "${SLOW_NODES}" ] && SLOW_LIST=$(sort -u "${SLOW_NODES}" | paste -sd, -)
if [ -n "${ECC_LIST}" ]; then echo "  hardware/ECC : ${ECC_LIST}"; fi
if [ -n "${SLOW_LIST}" ]; then echo "  desempenho   : ${SLOW_LIST}"; fi
ALL=$(printf '%s\n%s\n' "${ECC_LIST}" "${SLOW_LIST}" | tr ',' '\n' | grep -E 'sdumont[0-9]+' | sort -u | paste -sd, -)
if [ -n "${ALL}" ]; then
    echo
    echo "  Junte ao --exclude= existente:"
    echo "    ${ALL}"
else
    echo "  Nenhum nó isolável — variação pode ser sistêmica (rede/jitter)."
fi
rm -f "${SLOW_NODES}"
