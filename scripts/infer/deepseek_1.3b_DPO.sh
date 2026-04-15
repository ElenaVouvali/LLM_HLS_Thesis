ROOT=/home/ubuntu/runs/random_design_split_deepseek_1.3b
SCRIPT=/home/ubuntu/src/train_DPO_harp_xattn.py
SFT_SCRIPT=/home/ubuntu/src/train_SFT_xattn.py
DATA=/home/ubuntu/LLM_data/all_kernels_llm_data_multi_target.jsonl
MEM=/home/ubuntu/save/harp/memory_tokens
MODEL=deepseek-ai/deepseek-coder-1.3b-base
SPLIT_JSON=${ROOT}/splits/random_design_split.json
GPU=${GPU:-0}

run_dpo () {
  local TAG="$1"
  local OBJ="$2"

  CUDA_VISIBLE_DEVICES=${GPU} python -u "${SCRIPT}" \
    --dataset "${DATA}" \
    --memory_dir "${MEM}" \
    --model "${MODEL}" \
    --sft_script "${SFT_SCRIPT}" \
    --objective "${OBJ}" \
    --stage1_adapter_dir "${ROOT}/${TAG}_stage1/best_custom_stage1" \
    --stage2_harp_xattn_path "${ROOT}/${TAG}_stage2/best_custom_stage2/harp_xattn.pt" \
    --split_mode random_design \
    --split_json "${SPLIT_JSON}" \
    --output_dir "${ROOT}/${TAG}_stage3" \
    --train_xattn_dpo \
    --train_attn_gate_dpo \
    --max_length 7168 \
    --batch_size 1 \
    --grad_accum 4 \
    --epochs 3 \
    --eval_steps 400 \
    --save_steps 400 \
    --logging_steps 10 \
    --gradient_checkpointing \
    --lr_xattn 1e-5 \
    --lr_gate 5e-5 \
    --mem_dim 32 \
    --max_slots 64 \
    --every_n_layers 8 \
    --xattn_heads 4 \
    --xattn_dim_head 64 \
    --xattn_ff_mult 1 \
    --seed 123
}

run_dpo latency_extreme PARETO_LATENCY_EXTREME
run_dpo pareto_knee PARETO_KNEE
run_dpo area_extreme PARETO_AREA_EXTREME
