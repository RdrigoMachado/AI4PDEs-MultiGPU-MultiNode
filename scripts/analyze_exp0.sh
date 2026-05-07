#!/bin/bash
# Analisa resultados do Exp 0 — validação numérica + speedup.
# PAPER_CONTEXT, Seção 0.
#
# Uso:
#   bash scripts/analyze_exp0.sh
#
# Override (opcional):
#   REF_STRATEGY=blocking TEST_STRATEGY=async_b bash scripts/analyze_exp0.sh
#   RTOL=1e-5 ATOL=1e-5 ...

set -u

SCRATCH_DIR=${SCRATCH_DIR:-/scratch/g-assimila/rodrigo.machado2}
LOGS_DIR=${LOGS_DIR:-${SCRATCH_DIR}/logs_comm}
CSV_FILE=${CSV_FILE:-${LOGS_DIR}/exp0_results.csv}
RUNS_DIR=${RUNS_DIR:-${LOGS_DIR}/exp0_runs}

REF_STRATEGY=${REF_STRATEGY:-blocking}
TEST_STRATEGY=${TEST_STRATEGY:-async_b}
RTOL=${RTOL:-1e-5}
ATOL=${ATOL:-1e-5}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE="${REPO_DIR}/tools/compare_halo_outputs.py"

if [ ! -f "${COMPARE}" ]; then
    echo "ERRO: ${COMPARE} não encontrado." >&2
    exit 2
fi
if [ ! -d "${RUNS_DIR}" ]; then
    echo "ERRO: ${RUNS_DIR} não existe." >&2
    exit 2
fi

echo "================================================================"
echo "Exp 0 — análise"
echo "  Runs dir:  ${RUNS_DIR}"
echo "  CSV:       ${CSV_FILE}"
echo "  Ref:       ${REF_STRATEGY}"
echo "  Test:      ${TEST_STRATEGY}"
echo "  Tol:       rtol=${RTOL}  atol=${ATOL}"
echo "================================================================"

# -----------------------------------------------------------------
# Inventário das pastas
# -----------------------------------------------------------------
echo
echo "--- Inventário (#arquivos .npy por run) ---"
for d in "${RUNS_DIR}"/*/; do
    n=$(ls "${d}/FPS" 2>/dev/null | wc -l)
    printf "  %-40s %3d .npy\n" "$(basename "${d}")" "${n}"
done

# -----------------------------------------------------------------
# Validação numérica par-a-par
# -----------------------------------------------------------------
echo
echo "--- Validação numérica: ${REF_STRATEGY} vs ${TEST_STRATEGY} ---"
n_pass=0
n_fail=0
n_skip=0
for ref_dir in "${RUNS_DIR}/${REF_STRATEGY}_r"*; do
    [ -d "${ref_dir}" ] || continue
    base=$(basename "${ref_dir}")
    run_id=$(echo "${base}" | sed -E "s/^${REF_STRATEGY}_r([0-9]+)_.*/\1/")

    test_dir=$(ls -d "${RUNS_DIR}/${TEST_STRATEGY}_r${run_id}_"* 2>/dev/null | tail -1)
    if [ -z "${test_dir}" ]; then
        echo "  run ${run_id}: SEM PAR (${TEST_STRATEGY}_r${run_id}_*)"
        n_skip=$((n_skip + 1))
        continue
    fi
    if [ ! -d "${ref_dir}/FPS" ] || [ ! -d "${test_dir}/FPS" ]; then
        echo "  run ${run_id}: FPS ausente"
        n_skip=$((n_skip + 1))
        continue
    fi

    echo
    echo "  run ${run_id}:"
    echo "    ref:  $(basename "${ref_dir}")"
    echo "    test: $(basename "${test_dir}")"
    if python3 "${COMPARE}" --quiet --rtol "${RTOL}" --atol "${ATOL}" \
            "${ref_dir}/FPS" "${test_dir}/FPS" | sed 's/^/    /'; then
        n_pass=$((n_pass + 1))
    else
        n_fail=$((n_fail + 1))
    fi
done

echo
echo "  Resumo numérico: PASS=${n_pass}  FAIL=${n_fail}  SKIP=${n_skip}"

# -----------------------------------------------------------------
# Estatísticas de tempo a partir do CSV
# -----------------------------------------------------------------
echo
echo "--- Estatísticas de tempo (CSV) ---"
if [ ! -f "${CSV_FILE}" ]; then
    echo "  CSV não encontrado em ${CSV_FILE} — pulando."
else
    awk -F',' -v ref="${REF_STRATEGY}" -v tst="${TEST_STRATEGY}" '
        NR==1 {
            for (i=1;i<=NF;i++) col[$i]=i
            next
        }
        $col["STATUS"]=="SUCESSO" {
            s=$col["HALO_STRATEGY"]
            t=$col["TIME_S"] + 0
            sum[s]+=t
            ssq[s]+=t*t
            n[s]++
            if (!(s in tmin) || t<tmin[s]) tmin[s]=t
            if (!(s in tmax) || t>tmax[s]) tmax[s]=t
        }
        END {
            printf "  %-12s %3s %9s %9s %9s %9s\n", "strategy", "n", "mean_s", "std_s", "min_s", "max_s"
            for (s in sum) {
                m=sum[s]/n[s]
                v=ssq[s]/n[s]-m*m
                sd=(v>0?sqrt(v):0)
                printf "  %-12s %3d %9.2f %9.2f %9.2f %9.2f\n", s, n[s], m, sd, tmin[s], tmax[s]
            }
            if ((ref in sum) && (tst in sum)) {
                mr=sum[ref]/n[ref]
                mt=sum[tst]/n[tst]
                printf "\n  Speedup %s / %s: %.3fx  (redução %.1f%%)\n", ref, tst, mr/mt, (1-mt/mr)*100
            }
        }
    ' "${CSV_FILE}"
fi

# -----------------------------------------------------------------
# Veredito (critério Seção 0)
# -----------------------------------------------------------------
echo
echo "--- Veredito (critério Seção 0 do PAPER_CONTEXT) ---"
if [ "${n_fail}" -gt 0 ]; then
    echo "  ${TEST_STRATEGY}: NÃO passa numericamente — investigar antes de seguir."
elif [ "${n_pass}" -eq 0 ]; then
    echo "  ${TEST_STRATEGY}: nenhum par comparável; veredito inconclusivo."
else
    if [ -f "${CSV_FILE}" ]; then
        gain=$(awk -F',' -v ref="${REF_STRATEGY}" -v tst="${TEST_STRATEGY}" '
            NR==1 { for (i=1;i<=NF;i++) col[$i]=i; next }
            $col["STATUS"]=="SUCESSO" { s=$col["HALO_STRATEGY"]; sum[s]+=$col["TIME_S"]+0; n[s]++ }
            END {
                if ((ref in sum) && (tst in sum)) {
                    mr=sum[ref]/n[ref]; mt=sum[tst]/n[tst]
                    printf "%.2f", (1-mt/mr)*100
                }
            }
        ' "${CSV_FILE}")
        echo "  ${TEST_STRATEGY}: PASSA numericamente em ${n_pass} run(s)."
        echo "  Ganho de tempo: ${gain}%"
        gain_int=$(awk -v g="${gain}" 'BEGIN { printf "%d", g+0 }')
        if [ "${gain_int}" -ge 5 ]; then
            echo "  -> Recomendação: usar ${TEST_STRATEGY} como padrão para Exp 1/2."
        else
            echo "  -> Recomendação: manter ${REF_STRATEGY} (ganho <5%)."
        fi
    fi
fi
echo
