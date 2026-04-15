set -euo pipefail

ROOT=/home/ubuntu/runs/random_design_split_every_16_layers
OUT=/home/ubuntu/GAN/final_stage3_predictions_every_16_layers
SCRIPT=/home/ubuntu/src/eval_stage1_stage2_stage3.py
MODEL=deepseek-ai/deepseek-coder-7b-base

GAN_CODE=/home/ubuntu/GAN/network_placeholders.cpp

# IMPORTANT:
# Use a REAL memory pack here, not the broken self-symlink.
GAN_MEM_ORIG=/home/ubuntu/GAN/GAN.memory.pt

# Use a clean directory that contains ONLY the memory file for this inference run.
GAN_MEM_DIR=/home/ubuntu/GAN/memory_bank_stage3

# kernel_name must match the memory filename stem
KERNEL_NAME=GAN

mkdir -p "${OUT}/cases" "${OUT}/preds"
rm -rf "${GAN_MEM_DIR}"
mkdir -p "${GAN_MEM_DIR}"

# copy, do NOT symlink
cp -f "${GAN_MEM_ORIG}" "${GAN_MEM_DIR}/${KERNEL_NAME}.memory.pt"

make_case () {
  local OBJECTIVE="$1"
  local OUT_JSONL="$2"

  python - "$GAN_CODE" "$KERNEL_NAME" "$OBJECTIVE" "$OUT_JSONL" <<'PY'
import json, sys, pathlib

code_path, kernel_name, objective, out_jsonl = sys.argv[1:5]
code = pathlib.Path(code_path).read_text(encoding="utf-8")

row = {
    "kernel_name": kernel_name,
    "code": code,
    "obj_mode": objective,
    "objective": objective
}

with open(out_jsonl, "w", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote 1 row -> {out_jsonl}")
PY
}

run_stage3_one () {
  local TAG="$1"
  local OBJECTIVE_NAME="$2"

  local STAGE1_ADAPTER="${ROOT}/${TAG}_stage1/best_custom_stage1"
  local STAGE3_HARP="${ROOT}/${TAG}_stage3/best_custom_stage3/harp_xattn.pt"
  local TRAIN_SELECTED="${ROOT}/${TAG}_stage3/selected_debug/train_selected.jsonl"

  local CASES_JSONL="${OUT}/cases/${TAG}_gan.jsonl"
  local PREDS_JSONL="${OUT}/preds/${TAG}_gan_stage3_predictions.jsonl"

  make_case "${OBJECTIVE_NAME}" "${CASES_JSONL}"

  python -u "${SCRIPT}" \
    --stage stage3 \
    --model "${MODEL}" \
    --adapter_dir "${STAGE1_ADAPTER}" \
    --harp_xattn_path "${STAGE3_HARP}" \
    --memory_dir "${GAN_MEM_DIR}" \
    --input_jsonl "${CASES_JSONL}" \
    --candidate_bank_dataset "${TRAIN_SELECTED}" \
    --max_prompt_tokens 4096 \
    --score_reduction mean \
    --candidate_max_prefix_tokens 1536 \
    --candidate_keep_head_tokens 256 \
    --mem_dim 32 \
    --max_slots 64 \
    --every_n_layers 8 \
    --xattn_heads 4 \
    --xattn_dim_head 64 \
    --xattn_ff_mult 1 \
    --output_jsonl "${PREDS_JSONL}"
}

run_stage3_one latency_extreme PARETO_LATENCY_EXTREME
run_stage3_one pareto_knee PARETO_KNEE
run_stage3_one area_extreme PARETO_AREA_EXTREME

cat \
  "${OUT}/preds/latency_extreme_gan_stage3_predictions.jsonl" \
  "${OUT}/preds/pareto_knee_gan_stage3_predictions.jsonl" \
  "${OUT}/preds/area_extreme_gan_stage3_predictions.jsonl" \
  > "${OUT}/gan_stage3_all_objectives_verbose_1.3b.jsonl"

python - "${OUT}/gan_stage3_all_objectives_verbose_1.3b.jsonl" \
          "${OUT}/gan_stage3_all_objectives_for_hls_1.3b.jsonl" <<'PY'
import json, sys

inp, out = sys.argv[1], sys.argv[2]

with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        cand = row["candidates"][0]
        out_row = {
            "kernel_name": row["kernel_name"],
            "obj_mode": row["obj_mode"],
            "prediction": cand["canonical_prediction"],
        }
        g.write(json.dumps(out_row, ensure_ascii=False) + "\n")

print(f"Wrote synthesis-ready file -> {out}")
PY

wc -l "${OUT}/gan_stage3_all_objectives_for_hls_every_16_layers.jsonl"
