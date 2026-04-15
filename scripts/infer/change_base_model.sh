#!/usr/bin/env bash
set -u
set -o pipefail

# ============================================================
# Random-design-split retrain with CodeLlama-7b-hf
# Keeps the same pipeline shape; swaps only the backbone + run root.
#
# Recommended workflow:
#   1) RUN_STAGE3=0 bash this_script.sh
#   2) RUN_STAGE3=1 DPO_MAX_STEPS=1 bash this_script.sh
#      -> inspect pair_debug in each *_stage3 folder
#   3) RUN_STAGE3=1 DPO_MAX_STEPS=-1 bash this_script.sh
# ============================================================

GPU="${GPU:-0}"
SFT_SCRIPT="src/train_SFT_xattn.py"
DPO_SCRIPT="src/train_DPO_harp_xattn.py"

DATASET="/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl"
MEMORY_DIR="/home/ubuntu/save/harp/memory_tokens/"
MODEL="deepseek-ai/deepseek-coder-7b-base"

SEED=123
RUN_ROOT="/home/ubuntu/runs/random_design_split"
SPLIT_JSON="${RUN_ROOT}/splits/random_design_split.json"

RUN_STAGE1="${RUN_STAGE1:-1}"          # 1 = run stage1+stage2, 0 = skip stage1 and run stage2 only
RUN_STAGE12="${RUN_STAGE12:-1}"        # 1 = run stage1/stage2 logic, 0 = skip both and reuse existing adapters
RUN_STAGE3="${RUN_STAGE3:-0}"          # 0 = skip DPO, 1 = run DPO
DPO_MAX_STEPS="${DPO_MAX_STEPS:--1}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/splits"

# ============================================================
# Common SFT args (stage1 + stage2)
# ============================================================
COMMON_SFT_ARGS=(
  --run_mode two_stage
  --dataset "${DATASET}"
  --memory_dir "${MEMORY_DIR}"
  --model "${MODEL}"

  # -------------------------
  # Random design-point split
  # -------------------------
  --split_mode random_design
  --val_ratio 0.10
  --test_ratio 0.10
  --split_seed "${SEED}"
  --stratify_by_kernel

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
  --eval_steps 160
  --save_steps 160
  --stage2_epochs 4
  --stage2_eval_steps 160
  --stage2_save_steps 160
  --stage2_lr_xattn 1e-4
  --stage2_lr_gate 2e-4
  --stage2_lr_ff 0.0
  --stage2_lr_gate_ff 0.0

  # -------------------------
  # Reproducibility
  # -------------------------
  --seed "${SEED}"
)

# ============================================================
# Common DPO args (stage3)
# ============================================================
COMMON_DPO_ARGS=(
  --dataset "${DATASET}"
  --memory_dir "${MEMORY_DIR}"
  --model "${MODEL}"
  --sft_script "${SFT_SCRIPT}"

  --split_json "${SPLIT_JSON}"
  --split_mode random_design

  --top_k 8
  --goal_domination_penalty 0.25
  --goal_max_dominated_gap 0.12
  --score_weight_min 0.6
  --score_weight_power 1.0

  --dpo_chosen_top_k 4
  --dpo_hard_window 4
  --dpo_hard_negatives_per_chosen 2
  --dpo_medium_negatives_per_chosen 2
  --dpo_min_score_gap 0.02
  --dpo_hard_gap_max 0.14
  --dpo_medium_gap_max 0.30
  --dpo_min_primary_rel_gain 0.05
  --dpo_min_edit_distance 1
  --dpo_min_edit_frac 0.03
  --dpo_max_edit_frac 0.60

  --min_supervised_sites 2
  --min_site_coverage 0.85
  --selection_num_val_kernels 6
  --require_same_supervised_schema

  --value_loss_weight 1.0
  --beta 0.1
  --label_smoothing 0.0
  --sft_alpha 0.0

  --max_length 4096
  --batch_size 1
  --grad_accum 4
  --epochs 3
  --max_steps "${DPO_MAX_STEPS}"
  --eval_steps 105
  --save_steps 105
  --logging_steps 10
  --num_workers 4
  --group_by_length
  --gradient_checkpointing

  --lr_lora 0.0
  --lr_xattn 5e-5
  --lr_gate 2e-5
  --lr_ff 0.0
  --lr_gate_ff 0.0
  --lr_embed 0.0

  --train_xattn_dpo
  --train_attn_gate_dpo

  --mem_dim 32
  --max_slots 64
  --every_n_layers 8
  --xattn_heads 4
  --xattn_dim_head 64
  --xattn_ff_mult 1

  --save_total_limit 2
  --seed "${SEED}"
)

FAILURES=()

run_two_stage_goal () {
  local GOAL="$1"
  local TAG="$2"
  local SPLIT_FLAG="$3"   # --save_split_json or --split_json

  local LOG_FILE="${RUN_ROOT}/logs/${TAG}_sft_stage12.log"
  local STAGE1_DIR="${RUN_ROOT}/${TAG}_stage1"
  local STAGE2_DIR="${RUN_ROOT}/${TAG}_stage2"

  echo
  echo "===================================================================================================="
  echo "[START SFT 2-STAGE] ${GOAL}"
  echo "[START] stage1=${STAGE1_DIR}"
  echo "[START] stage2=${STAGE2_DIR}"
  echo "[START] log=${LOG_FILE}"
  echo "===================================================================================================="
  echo

  if ! CUDA_VISIBLE_DEVICES="${GPU}" python -u "${SFT_SCRIPT}" \
      "${COMMON_SFT_ARGS[@]}" \
      --objective "${GOAL}" \
      "${SPLIT_FLAG}" "${SPLIT_JSON}" \
      --stage1_output_dir "${STAGE1_DIR}" \
      --stage2_output_dir "${STAGE2_DIR}" \
      2>&1 | tee "${LOG_FILE}"; then
    echo "[FAIL SFT] ${GOAL}" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("SFT:${GOAL}")
    return 1
  fi

  echo "[OK SFT] ${GOAL}" | tee -a "${RUN_ROOT}/logs/_summary.log"
  return 0
}


run_stage2_only_goal () {
  local GOAL="$1"
  local TAG="$2"

  local LOG_FILE="${RUN_ROOT}/logs/${TAG}_sft_stage2_only.log"
  local STAGE1_BEST="${RUN_ROOT}/${TAG}_stage1/best_custom_stage1"
  local STAGE2_DIR="${RUN_ROOT}/${TAG}_stage2"

  echo
  echo "===================================================================================================="
  echo "[START SFT STAGE2-ONLY] ${GOAL}"
  echo "[START] init_adapter_dir=${STAGE1_BEST}"
  echo "[START] stage2=${STAGE2_DIR}"
  echo "[START] split_json=${SPLIT_JSON}"
  echo "[START] log=${LOG_FILE}"
  echo "===================================================================================================="
  echo

  if [ ! -d "${STAGE1_BEST}" ]; then
    echo "[FAIL STAGE2-ONLY] Missing stage1 adapter: ${STAGE1_BEST}" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("STAGE2_ONLY:${GOAL}")
    return 1
  fi

  if [ ! -f "${SPLIT_JSON}" ]; then
    echo "[FAIL STAGE2-ONLY] Missing split file: ${SPLIT_JSON}" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("STAGE2_ONLY:${GOAL}")
    return 1
  fi

  if ! CUDA_VISIBLE_DEVICES="${GPU}" python -u "${SFT_SCRIPT}" \
      --run_mode single \
      --dataset "${DATASET}" \
      --memory_dir "${MEMORY_DIR}" \
      --model "${MODEL}" \
      --objective "${GOAL}" \
      --split_json "${SPLIT_JSON}" \
      --split_mode random_design \
      --val_ratio 0.10 \
      --test_ratio 0.10 \
      --split_seed "${SEED}" \
      --stratify_by_kernel \
      --top_k 6 \
      --goal_domination_penalty 0.25 \
      --goal_max_dominated_gap 0.12 \
      --score_weight_min 0.6 \
      --score_weight_power 1.0 \
      --candidate_loss_weight 0.0 \
      --candidate_sites_per_sample 2 \
      --candidate_negatives_per_site 2 \
      --candidate_max_prefix_tokens 1536 \
      --candidate_keep_head_tokens 256 \
      --min_supervised_sites 2 \
      --min_site_coverage 0.85 \
      --selection_num_val_kernels 6 \
      --max_length 4096 \
      --epochs 4 \
      --batch_size 2 \
      --grad_accum 4 \
      --num_workers 4 \
      --group_by_length \
      --gradient_checkpointing \
      --lr_lora 0.0 \
      --lr_embed 0.0 \
      --lora_r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05 \
      --mem_dim 32 \
      --max_slots 64 \
      --every_n_layers 8 \
      --xattn_heads 4 \
      --xattn_dim_head 64 \
      --xattn_ff_mult 1 \
      --eval_steps 160 \
      --save_steps 160 \
      --loss_chunk_t 256 \
      --seed "${SEED}" \
      --lr_xattn 1e-4 \
      --lr_gate 2e-4 \
      --lr_ff 0.0 \
      --lr_gate_ff 0.0 \
      --value_loss_weight 1.0 \
      --init_adapter_dir "${STAGE1_BEST}" \
      --output_dir "${STAGE2_DIR}" \
      --best_dir_name "best_custom_stage2" \
      2>&1 | tee "${LOG_FILE}"; then
    echo "[FAIL STAGE2-ONLY] ${GOAL}" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("STAGE2_ONLY:${GOAL}")
    return 1
  fi

  echo "[OK STAGE2-ONLY] ${GOAL}" | tee -a "${RUN_ROOT}/logs/_summary.log"
  return 0
}


run_dpo_goal () {
  local GOAL="$1"
  local TAG="$2"

  local LOG_FILE="${RUN_ROOT}/logs/${TAG}_stage3.log"
  local STAGE1_DIR="${RUN_ROOT}/${TAG}_stage1/best_custom_stage1"
  local STAGE2_HARP="${RUN_ROOT}/${TAG}_stage2/best_custom_stage2/harp_xattn.pt"
  local STAGE3_DIR="${RUN_ROOT}/${TAG}_stage3"

  echo
  echo "===================================================================================================="
  echo "[START DPO] ${GOAL}"
  echo "[START] stage1_adapter=${STAGE1_DIR}"
  echo "[START] stage2_harp=${STAGE2_HARP}"
  echo "[START] stage3_out=${STAGE3_DIR}"
  echo "[START] DPO_MAX_STEPS=${DPO_MAX_STEPS}"
  echo "[START] log=${LOG_FILE}"
  echo "===================================================================================================="
  echo

  if ! CUDA_VISIBLE_DEVICES="${GPU}" python -u "${DPO_SCRIPT}" \
      "${COMMON_DPO_ARGS[@]}" \
      --objective "${GOAL}" \
      --stage1_adapter_dir "${STAGE1_DIR}" \
      --stage2_harp_xattn_path "${STAGE2_HARP}" \
      --output_dir "${STAGE3_DIR}" \
      2>&1 | tee "${LOG_FILE}"; then
    echo "[FAIL DPO] ${GOAL}" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("DPO:${GOAL}")
    return 1
  fi

  echo "[OK DPO] ${GOAL}" | tee -a "${RUN_ROOT}/logs/_summary.log"
  return 0
}


# ------------------------------------------------------------------
# Stage 1 + Stage 2, or Stage 2 only for all 3 objectives
# Stage 1 run creates and saves the random-design split manifest.
# ------------------------------------------------------------------
if [ "${RUN_STAGE12}" = "1" ]; then
  if [ "${RUN_STAGE1}" = "1" ]; then
    run_two_stage_goal "PARETO_LATENCY_EXTREME" "latency_extreme" "--save_split_json"
    run_two_stage_goal "PARETO_KNEE"            "pareto_knee"     "--split_json"
    run_two_stage_goal "PARETO_AREA_EXTREME"    "area_extreme"    "--split_json"
  else
    echo "[INFO] RUN_STAGE1=${RUN_STAGE1} -> skipping Stage 1 and reusing existing best_custom_stage1 adapters"
    run_stage2_only_goal "PARETO_LATENCY_EXTREME" "latency_extreme"
    run_stage2_only_goal "PARETO_KNEE"            "pareto_knee"
    run_stage2_only_goal "PARETO_AREA_EXTREME"    "area_extreme"
  fi
else
  echo "[INFO] RUN_STAGE12=${RUN_STAGE12} -> skipping Stage 1 and Stage 2, reusing existing adapters"
fi


# ------------------------------------------------------------------
# Stage 3 (optional)
# Recommended first pass:
#   RUN_STAGE3=1 DPO_MAX_STEPS=1 bash this_script.sh
# Then inspect:
#   ${RUN_ROOT}/*_stage3/pair_debug/
# If good:
#   RUN_STAGE3=1 DPO_MAX_STEPS=-1 bash this_script.sh
# ------------------------------------------------------------------
if [ "${RUN_STAGE3}" = "1" ]; then
  run_dpo_goal "PARETO_LATENCY_EXTREME" "latency_extreme"
  run_dpo_goal "PARETO_KNEE"            "pareto_knee"
  run_dpo_goal "PARETO_AREA_EXTREME"    "area_extreme"
else
  echo "[INFO] RUN_STAGE3=${RUN_STAGE3} -> skipping DPO stage3 for now"
fi

echo
echo "===================================================================================================="
echo "[DONE] Random-design-split (xattn every 16 layers) run finished"
echo "[ROOT] ${RUN_ROOT}"
echo "[SPLIT] ${SPLIT_JSON}"
if [ ${#FAILURES[@]} -eq 0 ]; then
  echo "[STATUS] All requested runs completed successfully"
else
  echo "[STATUS] Failures: ${FAILURES[*]}"
fi
echo "===================================================================================================="
