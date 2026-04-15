#!/usr/bin/env bash
set -u
set -o pipefail

GPU="${GPU:-0}"

SFT_SCRIPT="src/train_SFT_xattn.py"
DPO_SCRIPT="src/train_DPO_harp_xattn.py"

DATASET="/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl"
MEMORY_DIR="/home/ubuntu/save/harp/memory_tokens/"
MODEL="deepseek-ai/deepseek-coder-7b-base"

RUN_ROOT="${RUN_ROOT:-/home/ubuntu/runs/random_design_split}"
SEED=123
SPLIT_JSON="${RUN_ROOT}/splits/random_design_80_10_10_seed${SEED}.json"

mkdir -p "${RUN_ROOT}/logs"

# --------------------------------------------------------------------------------------------------
# Corrected stage2-only rerun config
# --------------------------------------------------------------------------------------------------
STAGE2_COMMON_ARGS=(
  --run_mode single
  --dataset "${DATASET}"
  --memory_dir "${MEMORY_DIR}"
  --model "${MODEL}"

  --split_json "${SPLIT_JSON}"
  --split_mode random_design
  --split_seed "${SEED}"
  --stratify_by_kernel

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

  --max_length 4096
  --epochs 4
  --batch_size 2
  --grad_accum 4
  --num_workers 4
  --group_by_length
  --gradient_checkpointing

  # stage2 behavior: freeze LoRA/embed, train only HARP xattn/gates
  --lr_lora 0.0
  --lr_embed 0.0
  --lr_xattn 1e-4
  --lr_gate 2e-4
  --lr_gate_ff 0.0
  --stage2_lr_ff 0.0

  --mem_dim 32
  --max_slots 64
  --every_n_layers 8
  --xattn_heads 4
  --xattn_dim_head 64
  --xattn_ff_mult 1

  --eval_steps 164
  --save_steps 164
  --best_dir_name best_custom_stage2

  --seed "${SEED}"
)

# --------------------------------------------------------------------------------------------------
# DPO config
# --------------------------------------------------------------------------------------------------
DPO_COMMON_ARGS=(
  --dataset "${DATASET}"
  --memory_dir "${MEMORY_DIR}"
  --model "${MODEL}"
  --sft_script "${SFT_SCRIPT}"

  --split_json "${SPLIT_JSON}"
  --split_mode random_design
  --split_seed "${SEED}"
  --stratify_by_kernel

  --top_k 6
  --goal_domination_penalty 0.25
  --goal_max_dominated_gap 0.12
  --score_weight_min 0.6
  --score_weight_power 1.0

  --dpo_chosen_top_k 3
  --dpo_hard_window 2
  --dpo_hard_negatives_per_chosen 2
  --dpo_medium_negatives_per_chosen 1
  --dpo_min_score_gap 0.03
  --dpo_hard_gap_max 0.15
  --dpo_medium_gap_max 0.35
  --dpo_min_primary_rel_gain 0.02
  --dpo_min_edit_distance 1
  --dpo_min_edit_frac 0.0
  --dpo_max_edit_frac 1.0

  --min_supervised_sites 2
  --min_site_coverage 0.85
  --selection_num_val_kernels 6
  --require_same_supervised_schema

  --beta 0.1
  --label_smoothing 0.0
  --sft_alpha 0.0

  --max_length 4096
  --batch_size 1
  --grad_accum 8
  --epochs 1
  --max_steps 60
  --eval_steps 60
  --save_steps 60
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

run_goal() {
  local GOAL="$1"
  local TAG="$2"

  local STAGE1_BEST_DIR="${RUN_ROOT}/${TAG}_stage1/best_custom_stage1"
  local STAGE2_FIXED_DIR="${RUN_ROOT}/${TAG}_stage2_fixed"
  local STAGE2_FIXED_CKPT="${STAGE2_FIXED_DIR}/checkpoint-164/harp_xattn.pt"
  local STAGE3_DIR="${RUN_ROOT}/${TAG}_stage3"

  local STAGE2_LOG="${RUN_ROOT}/logs/${TAG}_stage2_fixed.log"
  local STAGE3_LOG="${RUN_ROOT}/logs/${TAG}_stage3.log"

  echo
  echo "===================================================================================================="
  echo "[START] ${GOAL}"
  echo "[INFO] stage1 best adapter : ${STAGE1_BEST_DIR}"
  echo "[INFO] corrected stage2 dir: ${STAGE2_FIXED_DIR}"
  echo "[INFO] stage3 output dir   : ${STAGE3_DIR}"
  echo "===================================================================================================="
  echo

  if [ ! -d "${STAGE1_BEST_DIR}" ]; then
    echo "[FAIL] Missing stage1 best adapter: ${STAGE1_BEST_DIR}" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("${GOAL}:missing_stage1_best")
    return 1
  fi

  # ----------------------------------------------------------------------------------------------
  # Corrected stage2-only rerun on top of best_custom_stage1
  # ----------------------------------------------------------------------------------------------
  echo "[RUN] Corrected stage2 for ${GOAL}" | tee "${STAGE2_LOG}"

  if ! CUDA_VISIBLE_DEVICES="${GPU}" python -u "${SFT_SCRIPT}" \
      "${STAGE2_COMMON_ARGS[@]}" \
      --objective "${GOAL}" \
      --init_adapter_dir "${STAGE1_BEST_DIR}" \
      --output_dir "${STAGE2_FIXED_DIR}" \
      2>&1 | tee -a "${STAGE2_LOG}"; then
    echo "[FAIL] ${GOAL} corrected_stage2" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("${GOAL}:stage2_fixed")
    return 1
  fi

  if [ ! -f "${STAGE2_FIXED_CKPT}" ]; then
    echo "[FAIL] Missing corrected stage2 harp_xattn: ${STAGE2_FIXED_CKPT}" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("${GOAL}:missing_stage2_ckpt164")
    return 1
  fi

  # ----------------------------------------------------------------------------------------------
  # DPO on top of corrected stage2 HARP + same best_custom_stage1 LoRA prior
  # ----------------------------------------------------------------------------------------------
  echo "[RUN] Stage3 DPO for ${GOAL}" | tee "${STAGE3_LOG}"

  if ! CUDA_VISIBLE_DEVICES="${GPU}" python -u "${DPO_SCRIPT}" \
      "${DPO_COMMON_ARGS[@]}" \
      --objective "${GOAL}" \
      --stage1_adapter_dir "${STAGE1_BEST_DIR}" \
      --stage2_harp_xattn_path "${STAGE2_FIXED_CKPT}" \
      --output_dir "${STAGE3_DIR}" \
      2>&1 | tee -a "${STAGE3_LOG}"; then
    echo "[FAIL] ${GOAL} stage3_dpo" | tee -a "${RUN_ROOT}/logs/_summary.log"
    FAILURES+=("${GOAL}:stage3_dpo")
    return 1
  fi

  echo "[OK] ${GOAL}" | tee -a "${RUN_ROOT}/logs/_summary.log"
  return 0
}

run_goal "PARETO_LATENCY_EXTREME" "latency_extreme"
run_goal "PARETO_KNEE"            "pareto_knee"
run_goal "PARETO_AREA_EXTREME"    "area_extreme"

echo
echo "===================================================================================================="
echo "[DONE] Corrected stage2 + stage3 DPO pipeline finished"
echo "[ROOT] ${RUN_ROOT}"
if [ ${#FAILURES[@]} -eq 0 ]; then
  echo "[STATUS] All runs completed successfully"
else
  echo "[STATUS] Failures: ${FAILURES[*]}"
fi
echo "===================================================================================================="


