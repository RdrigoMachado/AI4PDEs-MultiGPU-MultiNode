#!/bin/bash
# Re-roda as células "lentas" do Strong scaling — os runs SUCESSO cujo EXEC_S
# foi inflado pela bimodalidade de clock/boost da GPU (ex.: a população ~75s do
# 4n contra a ~65s, ou ~48s vs ~33s no 8n). Diferente de resubmit_strong.sh
# (que só preenche células FALTANTES), este EVICTA as linhas lentas do CSV e as
# resubmete, de modo que a mediana passe a refletir o modo limpo dominante.
#
# Um run é "lento" se EXEC_S > min(EXEC_S do mesmo nº de nós) * (1 + THRESH).
# Default THRESH=0.05 (5%). O run mais rápido de cada nº de nós é sempre
# mantido (ele é o próprio min), então o script nunca esvazia uma célula.
#
# Rode antes scripts/diagnose_strong.sh: se houver nó genuinamente ruim, ele
# aparece em "SUSPEITO" e deve ir ao --exclude= de run_strong.sbatch ANTES do
# rerun. (Aqui o caso é temporal — mesmo nó roda rápido e lento — então re-rodar
# tem chance real de cair no modo rápido.)
#
# Uso (no SDumont):
#   bash scripts/rerun_slow_strong.sh --dry-run   # lista os lentos, não altera
#   bash scripts/rerun_slow_strong.sh             # evicta do CSV + resubmete
#
# Override via env (defaults shown):
#   THRESH=0.05 STEPS=40 PARTITION=sequana_gpu MAX_INFLIGHT=20 \
#     bash scripts/rerun_slow_strong.sh

set -u

SCRATCH_DIR="/scratch/g-assimila/rodrigo.machado2"
LOGS_DIR="${SCRATCH_DIR}/logs_comm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

THRESH=${THRESH:-0.05}
STEPS=${STEPS:-40}
PARTITION=${PARTITION:-sequana_gpu}
MAX_INFLIGHT=${MAX_INFLIGHT:-20}

# Grid FIXO (strong scaling) — idêntico a run_strong_all.sh / resubmit_strong.sh.
NX=256
NY=256
NZ=12288
TOPO=3d
IO=none

CSV_FILE="${LOGS_DIR}/strong_results.csv"
MANIFEST="${LOGS_DIR}/strong_manifest.tsv"

[ -f "${CSV_FILE}" ] || { echo "ERRO: ${CSV_FILE} não encontrado." >&2; exit 1; }

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

ts() { date +%H:%M:%S; }

walltime_for() {
    if [ "$1" -le 4 ]; then echo "00:08:00"; else echo "00:05:00"; fi
}

wait_for_slot() {
    while : ; do
        local inflight
        inflight=$(squeue --me -h -p "${PARTITION}" -o "%i" 2>/dev/null | wc -l)
        [ "${inflight}" -lt "${MAX_INFLIGHT}" ] && return 0
        sleep 30
    done
}

# Identifica linhas SUCESSO lentas. Emite "jobid nodes run exec_s" por linha.
slow_rows() {
    awk -F',' -v thr="${THRESH}" '
        $2=="SUCESSO" && ($11+0)==$11 {
            n=$5; e=$11
            if (!(n in mn) || e < mn[n]) mn[n]=e
            jid[NR]=$1; nd[NR]=$5; rn[NR]=$9; ex[NR]=$11
        }
        END {
            for (i=1;i<=NR;i++)
                if (i in jid && ex[i] > mn[nd[i]]*(1+thr))
                    print jid[i], nd[i], rn[i], ex[i]
        }
    ' "${CSV_FILE}"
}

mapfile -t SLOW < <(slow_rows)

if [ "${#SLOW[@]}" -eq 0 ]; then
    echo "Nenhum run lento acima de min*(1+${THRESH}). Nada a fazer."
    exit 0
fi

echo "Runs lentos (EXEC_S > min*(1+${THRESH})) — serão re-rodados:"
printf '  %-10s %-5s %-4s %s\n' JOB NÓS RUN EXEC_S
for line in "${SLOW[@]}"; do
    # shellcheck disable=SC2086
    set -- ${line}
    printf '  %-10s %-5s %-4s %s\n' "$1" "${2}n" "r$3" "$4"
done

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[dry-run] nenhuma alteração feita."
    exit 0
fi

# Backup e remoção das linhas lentas (por jobid).
BK="${CSV_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
cp "${CSV_FILE}" "${BK}"
echo "[$(ts)] backup do CSV: ${BK}"

EVICT=$(mktemp)
for line in "${SLOW[@]}"; do
    # shellcheck disable=SC2086
    set -- ${line}
    echo "$1"
done | sort -u > "${EVICT}"
n_evict=$(wc -l < "${EVICT}")
awk -F',' 'NR==FNR{drop[$1]=1; next} FNR==1 || !($1 in drop)' \
    "${EVICT}" "${CSV_FILE}" > "${CSV_FILE}.tmp"
mv "${CSV_FILE}.tmp" "${CSV_FILE}"
echo "[$(ts)] removidas ${n_evict} linhas lentas do CSV."
rm -f "${EVICT}"

# Resubmete cada célula (nodes, run) evictada.
for line in "${SLOW[@]}"; do
    # shellcheck disable=SC2086
    set -- ${line}
    nodes=$2; run=$3
    wt=$(walltime_for "${nodes}")
    wait_for_slot
    jobid=$(sbatch --parsable --nodes="${nodes}" --time="${wt}" -p "${PARTITION}" \
        "${SCRIPT_DIR}/run_strong.sbatch" \
        "${NX}" "${NY}" "${NZ}" "${run}" "${TOPO}" "${IO}" "${STEPS}" "${CSV_FILE}")
    if [ -z "${jobid}" ]; then
        echo "  [$(ts)] ERRO ao submeter ${nodes}n r${run}"
        continue
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -Iseconds)" "${jobid}" "${nodes}" "${TOPO}" "${IO}" \
        "${NX}" "${NY}" "${NZ}" "${run}" "${wt}" >> "${MANIFEST}"
    echo "  [$(ts)] re-submitted ${jobid}  ${nodes}n r${run}  (${wt})"
done

echo
echo "[$(ts)] done.  Track: squeue --me -p ${PARTITION} ; tail -f ${CSV_FILE}"
echo "Depois rode scripts/diagnose_strong.sh para confirmar que a dispersão sumiu."
