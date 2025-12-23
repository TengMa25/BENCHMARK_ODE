#!/usr/bin/env bash
set -euo pipefail

# =======================
# 0) Conda environment
# =======================
# 使用方式示例：
#   ENV_NAME=env_pysindy JOBS=4 ./scripts/run_pysindy_batch.sh
#   CONDA_BASE=$(conda info --base) ENV_NAME=env_pysindy ./scripts/run_pysindy_batch.sh
CONDA_BASE="${CONDA_BASE:-/home/mateng2025/anaconda3}"
ENV_NAME="${ENV_NAME:-env_pysindy}"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "[ERROR] conda init script not found:"
  echo "        ${CONDA_BASE}/etc/profile.d/conda.sh"
  echo "        Set CONDA_BASE correctly (try: conda info --base)."
  exit 2
fi

# 初始化 conda 并激活环境
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 强制使用该环境的 python，避免并行子进程漂移到 base/system python
PY_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  echo "[ERROR] Env python not executable: ${PY_BIN}"
  echo "        Check ENV_NAME=${ENV_NAME} and CONDA_BASE=${CONDA_BASE}"
  exit 2
fi

echo "[INFO] ENV_NAME=${ENV_NAME}"
echo "[INFO] PY_BIN=${PY_BIN}"
"${PY_BIN}" -c "import sys; print('[INFO] sys.executable:', sys.executable)"

# 预检查：避免批量跑到一半才发现没装包
"${PY_BIN}" - << 'PY'
import pysindy, numpy, sklearn
print("[INFO] pysindy:", pysindy.__version__)
print("[INFO] numpy:", numpy.__version__)
print("[INFO] sklearn:", sklearn.__version__)
PY

# =======================
# 1) Benchmark config
# =======================
REPO_ROOT="${REPO_ROOT:-/home/mateng2025/BENCHMARK_ODE}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"

SYSTEM="${SYSTEM:-vanderpol/demo}"
DT="${DT:-0.1}"

# variants（需要你的 runner 支持 --variant）
VARIANTS=(${VARIANTS:-"stlsq sr ensemble"})

CASE_FROM="${CASE_FROM:-0}"
CASE_TO="${CASE_TO:-9}"
DS_FROM="${DS_FROM:-1}"
DS_TO="${DS_TO:-10}"

JOBS="${JOBS:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/results/raw/${RUN_ID}/pysindy}"

export PYTHONPATH="${REPO_ROOT}"
export RUN_ID
export PY_BIN REPO_ROOT DATA_ROOT SYSTEM DT RUN_ID OUT_BASE PYTHONPATH

mkdir -p "${OUT_BASE}"

# =======================
# 2) One task
# =======================
task_one() {
  local variant="$1"
  local case_id="$2"
  local ds_id="$3"

  local out_dir="${OUT_BASE}/${variant}/${SYSTEM}/case_${case_id}"
  local out_jsonl="${out_dir}/ds_${ds_id}.jsonl"
  local out_coef="${out_dir}/ds_${ds_id}_coef.npy"
  mkdir -p "${out_dir}"

  "${PY_BIN}" -m runners.pysindy.run \
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
export -f task_one

# =======================
# 3) Generate tasks
# =======================
TASKS_FILE="$(mktemp)"
for v in "${VARIANTS[@]}"; do
  for c in $(seq -w "${CASE_FROM}" "${CASE_TO}"); do
    for d in $(seq -w "${DS_FROM}" "${DS_TO}"); do
      echo "${v} ${c} ${d}" >> "${TASKS_FILE}"
    done
  done
done

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] SYSTEM=${SYSTEM}"
echo "[INFO] VARIANTS=${VARIANTS[*]}"
echo "[INFO] CASES=$(seq -w "${CASE_FROM}" "${CASE_TO}" | tr '\n' ' ' | sed 's/ $//')"
echo "[INFO] DATASETS=$(seq -w "${DS_FROM}" "${DS_TO}" | tr '\n' ' ' | sed 's/ $//')"
echo "[INFO] DT=${DT}"
echo "[INFO] JOBS=${JOBS}"
echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] OUT_BASE=${OUT_BASE}"
echo "[INFO] Total tasks=$(wc -l < "${TASKS_FILE}")"

# =======================
# 4) Run tasks (parallel)
# =======================
cat "${TASKS_FILE}" | xargs -n 3 -P "${JOBS}" bash -lc 'task_one "$0" "$1" "$2"'
rm -f "${TASKS_FILE}"

echo "[DONE] Finished. Results under: ${OUT_BASE}"
SH

