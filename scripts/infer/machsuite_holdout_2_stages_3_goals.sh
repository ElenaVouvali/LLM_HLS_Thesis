#!/usr/bin/env bash
set -u
set -o pipefail

# =========================
# User-adjustable settings
# =========================
GPU="${GPU:-0}"
TRAIN_SCRIPT="src/train_SFT_xattn.py"

DATASET="/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl"
MEMORY_DIR="/home/ubuntu/save/harp/memory_tokens/"
MODEL="deepseek-ai/deepseek-coder-7b-base"

SEED=123
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/home/ubuntu/runs/unseen_machsuite_${STAMP}"
SPLIT_JSON="${RUN_ROOT}/splits/family_holdout_unseen_machsuite.json"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/splits"

# =========================
# Common arguments
# =========================
COMMON_ARGS=(
  --run_mode two_stage
  --dataset "${DATASET}"
  --memory_dir "${MEMORY_DIR}"
  --model "${MODEL}"

  # -------------------------
  # Family-level split
  # -------------------------
  --split_mode family
  --val_families "rodinia_pathfinder;spcl_example"
  --test_families "machsuite_sort_radix;machsuite_gemm;machsuite_md_knn;machsuite_spmv_ellpack;machsuite_viterbi;machsuite_stencil2d;machsuite_stencil3d"

  # -------------------------
  # Goal-aware point selection
  # -------------------------
  --top_k 6
  --goal_domination_penalty 0.25
  --goal_max_dominated_gap 0.12
  --score_weight_min 0.6
  --score_weight_power 1.0

  --candidate_loss_weight 0.0
  --candidate_sites_per_sample 2
  --candidate_negatives_per_site 2
  --candidate_max_prefix_tokens 1536
  --candidate_keep_head_tokens 256

  --min_supervised_sites 2
  --min_site_coverage 0.85
  --selection_num_val_kernels 6

  # -------------------------
  # Training
  # -------------------------
  --max_length 4096
  --epochs 4
  --batch_size 2
  --grad_accum 4
  --num_workers 4
  --group_by_length
  --gradient_checkpointing

  # -------------------------
  # LoRA
  # -------------------------
  --lr_lora 5e-5
  --lr_embed 5e-5
  --lora_r 8
  --lora_alpha 16
  --lora_dropout 0.05

  # -------------------------
  # HARP cross-attention
  # -------------------------
  --mem_dim 32
  --max_slots 64
  --every_n_layers 8
  --xattn_heads 4
  --xattn_dim_head 64
  --xattn_ff_mult 1

  # -------------------------
  # Checkpoint / eval cadence
  # -------------------------
  --eval_steps 120
  --save_steps 120
  --stage2_epochs 4
  --stage2_eval_steps 120
  --stage2_save_steps 120
  --stage2_lr_xattn 1e-4
  --stage2_lr_gate 2e-4
  --stage2_lr_ff 0.0
  --stage2_lr_gate_ff 0.0

  # -------------------------
  # Reproducibility
  # -------------------------
  --seed "${SEED}"
)

FAILURES=()

run_goal() {
  local GOAL="$1"
  local TAG="$2"
  local SPLIT_MODE_FLAG="$3"   # either --save_split_json or --split_json

  local LOG_FILE="${RUN_ROOT}/logs/${TAG}.log"
  local STAGE1_DIR="${RUN_ROOT}/${TAG}_stage1"
  local STAGE2_DIR="${RUN_ROOT}/${TAG}_stage2"

  echo
  echo "===================================================================================================="
  echo "[START] ${GOAL}"
  echo "[START] stage1=${STAGE1_DIR}"
  echo "[START] stage2=${STAGE2_DIR}"
  echo "[START] log=${LOG_FILE}"
  echo "===================================================================================================="
  echo

  if ! CUDA_VISIBLE_DEVICES="${GPU}" python -u "${TRAIN_SCRIPT}" \
      "${COMMON_ARGS[@]}" \
      --objective "${GOAL}" \
      "${SPLIT_MODE_FLAG}" "${SPLIT_JSON}" \
      --stage1_output_dir "${STAGE1_DIR}" \
      --stage2_output_dir "${STAGE2_DIR}" \
      2>&1 | tee "${LOG_FILE}"; then
    echo "[FAIL] ${GOAL}" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("${GOAL}")
    return 1
  fi

  echo "[OK] ${GOAL}" | tee -a "${RUN_ROOT}/logs/_summary.log"
  return 0
}

# First run creates and saves the raw family split manifest
run_goal "PARETO_LATENCY_EXTREME" "latency_extreme" "--save_split_json"

# Reuse the exact same split for the other two objectives
run_goal "PARETO_KNEE"         "pareto_knee"    "--split_json"
run_goal "PARETO_AREA_EXTREME" "area_extreme"   "--split_json"

echo
echo "===================================================================================================="
echo "[DONE] Unseen-MachSuite family-holdout run finished"
echo "[ROOT] ${RUN_ROOT}"
echo "[SPLIT] ${SPLIT_JSON}"
if [ ${#FAILURES[@]} -eq 0 ]; then
  echo "[STATUS] All runs completed successfully"
else
  echo "[STATUS] Failures: ${FAILURES[*]}"
fi
echo "===================================================================================================="
