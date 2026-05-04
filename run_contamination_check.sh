#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_contamination_check.sh
#
# Sharded-Likelihood contamination test for open-source models.
# Just edit the two sections marked "CHANGE THIS" below, then run:
#
#   bash run_contamination_check.sh
# ═════════════════════════════════════════════════════════════════════════════

set -euo pipefail

DEFAULT_MODEL_ROOT="${HOME}/models"
DEFAULT_DATASETS_DIR="${HOME}/8datasets"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CHANGE THIS 1 — Model                                                   ║
# ║                                                                           ║
# ║  Set MODEL_NAME and MODEL_PATH for the model you want to test.           ║
# ║                                                                           ║
# ║  Available models (ls /data/models/):                                    ║
# ║    Qwen3.5-2B  |  qwen3-4b  |  qwen3.5-9b  |  gemma3-4b                ║
# ║    internvl3.5-4b  |  kazllm-8b  |  (and others)                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
MODEL_ROOT="${MODEL_ROOT:-${DEFAULT_MODEL_ROOT}}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CHANGE THIS 2 — Datasets to check                                       ║
# ║                                                                           ║
# ║  Comment out any datasets you don't want to test.                        ║
# ║  To test all 8, leave everything uncommented.                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
DATASETS=(
    #mmlu_all            # Standard MMLU (English, 4 choices)
    #mmlu_cf_all         # MMLU counterfactual variant
    #mmlu_pro_all        # MMLU-Pro (up to 10 choices)
    #kazmmlu_all         # KazMMLU (Kazakh, 5 choices)
    #rummlu_all          # RuMMLU (Russian, 4 choices)
    #MMLU_KAZ_Translation  # MMLU translated to Kazakh
    MMLU_RUS_Translation  # MMLU translated to Russian
)


# ─────────────────────────────────────────────────────────────────────────────
# Advanced settings — only change these if you know what you're doing
# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_LEN=2048            # token context window length
STRIDE=1024                 # sliding window stride (half of context is typical)
NUM_SHARDS=50               # number of shards
PERMUTATIONS_PER_SHARD=100  # shuffled permutations per shard
NUM_GPUS="${NUM_GPUS:-1}"   # visible GPU count to use when multiple are available
RANDOM_SEED=0               # for reproducibility
MAX_EXAMPLES=5000           # max examples to use per dataset (0 = no limit)
DATASETS_DIR="${DATASETS_DIR:-${SHARDED_LIKELIHOOD_DATASETS_DIR:-${DEFAULT_DATASETS_DIR}}}"

# ─────────────────────────────────────────────────────────────────────────────
# Internal — do not edit below this line
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARED_DIR="${SCRIPT_DIR}/prepared_data"
RESULTS_DIR="${SCRIPT_DIR}/results"
MODEL_RESULTS_DIR="${RESULTS_DIR}/${MODEL_NAME}"

# ── Validate model path ───────────────────────────────────────────────────────
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] Model not found at: ${MODEL_PATH}"
    echo "        Set MODEL_NAME/MODEL_PATH before running this script."
    echo "        Current MODEL_ROOT: ${MODEL_ROOT}"
    if [ -d "${MODEL_ROOT}" ]; then
        echo "        Available models in ${MODEL_ROOT}:"
        ls "${MODEL_ROOT}" | sed 's/^/          /'
    else
        echo "        Model root does not exist yet. Create it or download a model there first."
    fi
    exit 1
fi

mkdir -p "${PREPARED_DIR}" "${RESULTS_DIR}" "${MODEL_RESULTS_DIR}"

# ── Step 1: Convert CSVs to JSONL ─────────────────────────────────────────────
echo "============================================================"
echo " Step 1: Preparing datasets"
echo "============================================================"
echo " Dataset source dir: ${DATASETS_DIR}"
for DATASET in "${DATASETS[@]}"; do
    JSONL="${PREPARED_DIR}/${DATASET}.jsonl"
    if [ -f "${JSONL}" ]; then
        echo "[SKIP] ${DATASET}.jsonl already exists. Delete it to re-convert."
    else
        echo "[CONVERT] ${DATASET}..."
        python3 "${SCRIPT_DIR}/prepare_datasets.py" --dataset "${DATASET}" --datasets_dir "${DATASETS_DIR}"
    fi
done
echo ""

# ── Step 2: Run contamination test on each dataset ───────────────────────────
echo "============================================================"
echo " Step 2: Running Sharded Likelihood test"
echo "   Model : ${MODEL_PATH}"
echo "   GPUs  : ${NUM_GPUS}"
echo "   Shards: ${NUM_SHARDS}  |  Perms/shard: ${PERMUTATIONS_PER_SHARD}"
echo "   Context len: ${CONTEXT_LEN}  |  Stride: ${STRIDE}"
echo "============================================================"

SUMMARY_FILE="${MODEL_RESULTS_DIR}/summary.txt"
echo "Model: ${MODEL_PATH}" > "${SUMMARY_FILE}"
echo "Run date: $(date)" >> "${SUMMARY_FILE}"
echo "------------------------------------------------" >> "${SUMMARY_FILE}"

for DATASET in "${DATASETS[@]}"; do
    JSONL="${PREPARED_DIR}/${DATASET}.jsonl"
    if [ ! -f "${JSONL}" ]; then
        echo "[SKIP] ${DATASET}: JSONL not found, skipping."
        continue
    fi

    LOG_FILE="${MODEL_RESULTS_DIR}/${DATASET}.json"
    echo ""
    echo "──────────────────────────────────────────────"
    echo " Dataset : ${DATASET}"
    echo " Output  : ${LOG_FILE}"
    echo "──────────────────────────────────────────────"

    python3 "${SCRIPT_DIR}/compute_sharded_comparison_test.py" \
        "${MODEL_PATH}" \
        "${JSONL}" \
        --context_len "${CONTEXT_LEN}" \
        --stride "${STRIDE}" \
        --num_shards "${NUM_SHARDS}" \
        --permutations_per_shard "${PERMUTATIONS_PER_SHARD}" \
        --num_gpus "${NUM_GPUS}" \
        --random_seed "${RANDOM_SEED}" \
        --max_examples "${MAX_EXAMPLES}" \
        --log_file_path "${LOG_FILE}"

    # Extract p-value, compute verdict, append to summary
    if [ -f "${LOG_FILE}" ]; then
        PVAL=$(python3 -c "import json; d=json.load(open('${LOG_FILE}')); print(f\"{d['pval']:.6f}\")" 2>/dev/null || echo "N/A")
        if python3 -c "exit(0 if float('${PVAL}') < 0.05 else 1)" 2>/dev/null; then
            VERDICT="CONTAMINATED (p < 0.05)"
        else
            VERDICT="CLEAN        (p >= 0.05)"
        fi
        LINE="${DATASET}: p-value = ${PVAL}  →  ${VERDICT}"
        echo "${LINE}" | tee -a "${SUMMARY_FILE}"
    fi
done


echo ""
echo "============================================================"
echo " Done! Summary saved to: ${SUMMARY_FILE}"
echo "============================================================"
cat "${SUMMARY_FILE}"
