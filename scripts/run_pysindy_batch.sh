#!/usr/bin/env bash
set -euo pipefail

# --------- user config ----------
REPO_ROOT="/home/mateng2025/BENCHMARK_ODE"
DATA_ROOT="${REPO_ROOT}/data"
SYSTEM="vanderpol/demo"
DT="0.1"

# 三个“方法/变体”
VARIANTS=("stlsq" "sr" "ensemble")

# case: 00..09, dataset: 01..10
CASE_FROM=0
CASE_TO=9
DS_FROM=1
DS_TO=10

# 并行度：1 = 串行；建议先用 2~4 试跑
JOBS="${JOBS:-2}"

# 输出 run_id（目录隔离）
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${REPO_ROOT}/results/raw/${RUN_ID}/pysindy"

# python module import path
export PYTHONPATH="${REPO_ROOT}"

# 可选：如果你要强制用某个 conda env，在外部先 conda activate env_pysindy
# --------------------------------

mkdir -p "${OUT_DIR}"

task_line() {
  local variant="$1"
  local case_id="$2"
  local ds_id="$3"
  local out_jsonl="${OUT_DIR}/${variant}/${SYSTEM}/case_${case_id}/ds_${ds_id}.jsonl"
  local out_coef="${OUT_DIR}/${variant}/${SYSTEM}/case_${case_id}/ds_${ds_id}_coef.npy"
  mkdir -p "$(dirname "${out_jsonl}")"

  # 注意：我们用 python -m 的方式，避免 import runners 失败
  python -m runners.pysindy.run \
    --method pysindy \
    --variant "${variant}" \
    --system "${SYSTEM}" \
    --case "${case_id}" \
    --dataset "${ds_id}" \
    --data_root "${DATA_ROOT}" \
    --dt "${DT}" \
    --rep 1 \
    --warmup 0 \
    --out "${out_jsonl}" \
    --coef_out "${out_coef}"
}

export -f task_line
export REPO_ROOT DATA_ROOT SYSTEM DT RUN_ID OUT_DIR PYTHONPATH

# 生成任务列表：variant case ds
TASKS_FILE="$(mktemp)"
for v in "${VARIANTS[@]}"; do
  for c in $(seq -w ${CASE_FROM} ${CASE_TO}); do
    for d in $(seq -w ${DS_FROM} ${DS_TO}); do
      echo "${v} ${c} ${d}" >> "${TASKS_FILE}"
    done
  done
done

echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] JOBS=${JOBS}"
echo "[INFO] Tasks=$(wc -l < "${TASKS_FILE}")  (variants=${#VARIANTS[@]}, cases=$((CASE_TO-CASE_FROM+1)), ds=$((DS_TO-DS_FROM+1)))"

# 并行执行（不依赖 GNU parallel）：xargs -P
# 如果你的系统 xargs 不支持 -P，请告诉我，我给你纯 bash 后台队列版本。
cat "${TASKS_FILE}" | xargs -n 3 -P "${JOBS}" bash -lc 'task_line "$0" "$1" "$2"' 

rm -f "${TASKS_FILE}"

echo "[DONE] Batch finished. Results under: ${OUT_DIR}"
SH

