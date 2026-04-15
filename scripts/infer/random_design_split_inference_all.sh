ROOT=/home/ubuntu/runs/random_design_split
OUT=/home/ubuntu/runs/random_design_split/final_stage3_test_batch
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

run_stage3_batch () {
  local TAG="$1"

  local STAGE1_ADAPTER="${ROOT}/${TAG}_stage1/best_custom_stage1"
  local STAGE3_HARP="${ROOT}/${TAG}_stage3/best_custom_stage3/harp_xattn.pt"
  local TEST_SELECTED="${ROOT}/${TAG}_stage3/selected_debug/test_selected.jsonl"
  local TRAIN_SELECTED="${ROOT}/${TAG}_stage3/selected_debug/train_selected.jsonl"
  local CASES_JSONL="${OUT}/cases/${TAG}_one_per_kernel.jsonl"
  local PREDS_JSONL="${OUT}/preds/${TAG}_stage3_predictions.jsonl"

  make_one_case_per_kernel "${TEST_SELECTED}" "${CASES_JSONL}"

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
    --output_jsonl "${PREDS_JSONL}"
}

run_stage3_batch latency_extreme
run_stage3_batch pareto_knee
run_stage3_batch area_extreme

cat \
  "${OUT}/preds/latency_extreme_stage3_predictions.jsonl" \
  "${OUT}/preds/pareto_knee_stage3_predictions.jsonl" \
  "${OUT}/preds/area_extreme_stage3_predictions.jsonl" \
  > "${OUT}/stage3_all_objectives_predictions_162_verbose.jsonl"

python - "${OUT}/stage3_all_objectives_predictions_162_verbose.jsonl" \
          "${OUT}/stage3_all_objectives_predictions_162_for_hls.jsonl" <<'PY'
import json, sys

inp, out = sys.argv[1], sys.argv[2]

with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        cand = row["candidates"][0]   # num_samples=1 default, so candidate 0 is the final prediction
        out_row = {
            "kernel_name": row["kernel_name"],
            "obj_mode": row["obj_mode"],
            "prediction": cand["canonical_prediction"],
        }
        g.write(json.dumps(out_row, ensure_ascii=False) + "\n")

print(f"Wrote synthesis-ready file -> {out}")
PY

wc -l "${OUT}/cases/"*_one_per_kernel.jsonl
wc -l "${OUT}/stage3_all_objectives_predictions_162_for_hls.jsonl"
