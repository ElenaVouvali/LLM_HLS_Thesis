#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/ubuntu/runs/serrano_holdout
OUT=/home/ubuntu/runs/serrano_holdout/final_ablation_all_stages
SCRIPT=/home/ubuntu/src/eval_stage1_stage2_stage3.py
MODEL=deepseek-ai/deepseek-coder-7b-base
MEMORY_DIR=/home/ubuntu/save/harp/memory_tokens

mkdir -p "${OUT}/cases" "${OUT}/preds"

make_one_case_per_kernel () {
  local IN_JSONL="$1"
  local OUT_JSONL="$2"

  python - "$IN_JSONL" "$OUT_JSONL" <<'PY'
import json, sys

inp, out = sys.argv[1], sys.argv[2]
best = {}

with open(inp, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        k = row["kernel_name"]
        key = (
            int(row.get("_rank_within_kernel", 10**9)),
            float(row.get("_score", 1e9)),
        )
        if k not in best or key < best[k][0]:
            best[k] = (key, row)

rows = [best[k][1] for k in sorted(best.keys())]

with open(out, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} rows -> {out}")
PY
}

run_ablation_for_objective () {
  local TAG="$1"

  # Use ONE common case file and ONE common candidate bank for all 3 stages of this objective
  local TEST_SELECTED="${ROOT}/${TAG}_stage3/selected_debug/test_selected.jsonl"
  local TRAIN_SELECTED="${ROOT}/${TAG}_stage3/selected_debug/train_selected.jsonl"
  local CASES_JSONL="${OUT}/cases/${TAG}_one_per_kernel.jsonl"

  local STAGE1_ADAPTER="${ROOT}/${TAG}_stage1/best_custom_stage1"
  local STAGE2_HARP="${ROOT}/${TAG}_stage2/best_custom_stage2/harp_xattn.pt"
  local STAGE3_HARP="${ROOT}/${TAG}_stage3/best_custom_stage3/harp_xattn.pt"

  make_one_case_per_kernel "${TEST_SELECTED}" "${CASES_JSONL}"

  # -------------------------
  # Stage 1
  # -------------------------
  python -u "${SCRIPT}" \
    --stage stage1 \
    --model "${MODEL}" \
    --adapter_dir "${STAGE1_ADAPTER}" \
    --input_jsonl "${CASES_JSONL}" \
    --candidate_bank_dataset "${TRAIN_SELECTED}" \
    --max_prompt_tokens 4096 \
    --score_reduction mean \
    --candidate_max_prefix_tokens 1536 \
    --candidate_keep_head_tokens 256 \
    --output_jsonl "${OUT}/preds/${TAG}_stage1_predictions.jsonl"

  # -------------------------
  # Stage 2
  # -------------------------
  python -u "${SCRIPT}" \
    --stage stage2 \
    --model "${MODEL}" \
    --adapter_dir "${STAGE1_ADAPTER}" \
    --harp_xattn_path "${STAGE2_HARP}" \
    --memory_dir "${MEMORY_DIR}" \
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
    --output_jsonl "${OUT}/preds/${TAG}_stage2_predictions.jsonl"

  # -------------------------
  # Stage 3
  # -------------------------
  python -u "${SCRIPT}" \
    --stage stage3 \
    --model "${MODEL}" \
    --adapter_dir "${STAGE1_ADAPTER}" \
    --harp_xattn_path "${STAGE3_HARP}" \
    --memory_dir "${MEMORY_DIR}" \
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
    --output_jsonl "${OUT}/preds/${TAG}_stage3_predictions.jsonl"
}

run_ablation_for_objective latency_extreme
run_ablation_for_objective pareto_knee
run_ablation_for_objective area_extreme

cat \
  "${OUT}/preds/latency_extreme_stage1_predictions.jsonl" \
  "${OUT}/preds/latency_extreme_stage2_predictions.jsonl" \
  "${OUT}/preds/latency_extreme_stage3_predictions.jsonl" \
  "${OUT}/preds/pareto_knee_stage1_predictions.jsonl" \
  "${OUT}/preds/pareto_knee_stage2_predictions.jsonl" \
  "${OUT}/preds/pareto_knee_stage3_predictions.jsonl" \
  "${OUT}/preds/area_extreme_stage1_predictions.jsonl" \
  "${OUT}/preds/area_extreme_stage2_predictions.jsonl" \
  "${OUT}/preds/area_extreme_stage3_predictions.jsonl" \
  > "${OUT}/all_stages_all_objectives_9_verbose.jsonl"

python - "${OUT}/all_stages_all_objectives_9_verbose.jsonl" \
          "${OUT}/all_stages_all_objectives_9_for_hls.jsonl" <<'PY'
import json, sys, os

inp, out = sys.argv[1], sys.argv[2]

def infer_stage_from_path(path):
    name = os.path.basename(path).lower()
    if "_stage1_" in name:
        return "stage1"
    if "_stage2_" in name:
        return "stage2"
    if "_stage3_" in name:
        return "stage3"
    return None

# build stage lookup from filename while reading line-by-line file groups is awkward,
# so instead just parse from each row's source file grouping by re-reading at higher level
# simpler approach: stage is stored externally below via filename order not row content.
PY
