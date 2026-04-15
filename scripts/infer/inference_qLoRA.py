# -----------------------------------------------------------
#                  Inference for LIFT-like model
# -----------------------------------------------------------

import os
import re
import json
import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# Paths / runtime
# ============================================================
GPU_ID = 0

MODEL_NAME = "deepseek-ai/deepseek-coder-7b-base"
OUTPUT_DIR = "/home/ubuntu/deepseek_7B_qLoRA_GNN_all_kernels_perf"
LORA_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")
TOKENIZER_DIR = os.path.join(OUTPUT_DIR, "tokenizer")

# Histogram input with placeholders
CODE_FILE = "/home/ubuntu/Histogram/xf_histogram_placeholders.hpp"

# Outputs
OUT_TXT = os.path.join(OUTPUT_DIR, "histogram_lift_prediction.txt")
OUT_JSON = os.path.join(OUTPUT_DIR, "histogram_prediction.json")

# Same context budget used in training
MAX_LENGTH = 3072
MAX_NEW_TOKENS_CAP = 2048


# ============================================================
# Prompt (copied from training)
# ============================================================
PROMPT_TEMPLATE = """
### Task
Given this HLS kernel with pragma placeholders of the form auto{{...}}, predict pragma values that minimize latency.

Output ONLY the filled directives, one per line, using EXACTLY the placeholders that appear in the kernel.

Use exactly this format:
auto{{_PIPE_LX}} = <non-negative integer>
auto{{_UNROLL_LX}} = <non-negative integer>
auto{{_ARRAY_T_LX}} = <block|cyclic|complete>
auto{{_ARRAY_F_LX}} = <non-negative integer>
auto{{_ARRAY_D_LX}} = <non-negative integer>

Do not add explanations, comments, or extra headings.

### Kernel
{code}

### Directives:
""".strip()


# ============================================================
# Parsing / post-processing helpers
# Aligned with your training-time LIFT code
# ============================================================
_ASSIGN_PAT = re.compile(
    r"""
    (?:auto)?
    \{
    ([_A-Z0-9]{3,40})
    \}
    \s*=\s*
    ([a-zA-Z0-9_]+)
    """,
    re.VERBOSE,
)

_PLACEHOLDER_PAT = re.compile(r"auto\{([^}]+)\}")
_ALLOWED_KEY_RE = re.compile(r"^_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+$")


def normalize_hls_val(val: str) -> str:
    val = str(val).lower().strip()
    if "complete" in val:
        return "complete"
    if "cycl" in val:
        return "cyclic"
    if "block" in val:
        return "block"

    m = re.search(r"\d+", val)
    if m:
        return str(int(m.group(0)))
    return val


def parse_assignments(text: str) -> dict:
    found = {}
    for m in _ASSIGN_PAT.finditer(text):
        key = m.group(1).strip().upper()
        val = normalize_hls_val(m.group(2))
        found[key] = val
    return found


def extract_placeholders(template_code: str) -> list:
    seen = set()
    ordered = []
    for m in _PLACEHOLDER_PAT.finditer(template_code):
        key = m.group(1).strip().upper()
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def build_fallback_map(raw_assigns: dict) -> dict:
    fallback = {}
    for k, v in raw_assigns.items():
        suffix_match = re.search(r"L\d+", k)
        if not suffix_match:
            continue
        suffix = suffix_match.group(0)

        if "PIPE" in k or "PIROLL" in k:
            fallback[f"PIPE_{suffix}"] = v
        elif "UNROLL" in k:
            fallback[f"UNROLL_{suffix}"] = v
        elif "ARRAY_T" in k:
            fallback[f"ARRAY_T_{suffix}"] = v
        elif "ARRAY_F" in k:
            fallback[f"ARRAY_F_{suffix}"] = v
        elif "ARRAY_D" in k:
            fallback[f"ARRAY_D_{suffix}"] = v
    return fallback


def coerce_value(key: str, value: str) -> str:
    if key.startswith("_ARRAY_T_"):
        v = str(value).lower().strip()
        if v in ("block", "cyclic", "complete"):
            return v
        v2 = normalize_hls_val(v)
        return v2 if v2 in ("block", "cyclic", "complete") else "block"

    v = str(value).strip()
    if v.isdigit():
        return v

    if key.startswith("_ARRAY_D_"):
        return "1"
    return "0"


def resolve_assigns_for_template(template_code: str, raw_assigns: dict) -> dict:
    raw = {k.strip().upper(): normalize_hls_val(v) for k, v in raw_assigns.items()}
    fallback = build_fallback_map(raw)

    resolved = {}
    for key in extract_placeholders(template_code):
        key = key.upper()

        if not _ALLOWED_KEY_RE.match(key):
            continue

        value = raw.get(key)

        if value is None:
            suffix_match = re.search(r"L\d+", key)
            if suffix_match:
                suffix = suffix_match.group(0)
                if key.startswith("_PIPE_"):
                    value = fallback.get(f"PIPE_{suffix}")
                elif key.startswith("_UNROLL_"):
                    value = fallback.get(f"UNROLL_{suffix}")
                elif key.startswith("_ARRAY_T_"):
                    value = fallback.get(f"ARRAY_T_{suffix}")
                elif key.startswith("_ARRAY_F_"):
                    value = fallback.get(f"ARRAY_F_{suffix}")
                elif key.startswith("_ARRAY_D_"):
                    value = fallback.get(f"ARRAY_D_{suffix}")

        if value is None:
            value = "0"

        resolved[key] = coerce_value(key, value)

    return resolved


def render_directives(template_code: str, resolved_assigns: dict) -> str:
    lines = []
    for key in extract_placeholders(template_code):
        if not _ALLOWED_KEY_RE.match(key):
            continue
        val = resolved_assigns[key]
        lines.append(f"auto{{{key}}} = {val}")
    return "\n".join(lines)


def count_placeholders(template_code: str) -> int:
    return sum(1 for key in extract_placeholders(template_code) if _ALLOWED_KEY_RE.match(key))


# ============================================================
# Model loading
# ============================================================
def load_tokenizer():
    tok_src = TOKENIZER_DIR if os.path.isdir(TOKENIZER_DIR) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model():
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)
        print("Using CUDA device:", GPU_ID, torch.cuda.get_device_name(GPU_ID))
    else:
        print("CUDA not available, running on CPU.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=torch.cuda.is_available(),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Keeps your script robust against the DeepSeek / transformers parallel_style issue
    if getattr(config, "parallel_style", None) is None:
        config.parallel_style = "none"

    device_map = {"": GPU_ID} if torch.cuda.is_available() else {"": "cpu"}

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = model.config.pad_token_id or base_model.config.pad_token_id
    model.config.eos_token_id = model.config.eos_token_id or base_model.config.eos_token_id
    return model


# ============================================================
# Inference
# ============================================================
def run_histogram_inference():
    if not os.path.isfile(CODE_FILE):
        raise FileNotFoundError(f"Histogram code file not found: {CODE_FILE}")

    with open(CODE_FILE, "r", encoding="utf-8") as f:
        code_str = f.read()

    tokenizer = load_tokenizer()
    model = load_model()

    eval_prompt = PROMPT_TEMPLATE.format(code=code_str)
    prompt_ids = tokenizer(eval_prompt, add_special_tokens=False)["input_ids"]

    # Same truncation philosophy as training: keep the tail
    if len(prompt_ids) > MAX_LENGTH:
        prompt_ids = prompt_ids[-MAX_LENGTH:]

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    num_placeholders = count_placeholders(code_str)
    max_new_tokens = min(MAX_NEW_TOKENS_CAP, max(128, 16 * num_placeholders))

    print(f"Prompt tokens: {len(prompt_ids)}")
    print(f"Pragma placeholders: {num_placeholders}")
    print(f"max_new_tokens: {max_new_tokens}")

    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # aligned with your current inference logic
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the continuation, not the prompt
    new_tokens = gen[0, input_ids.shape[1]:]
    raw_generation = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    raw_assigns = parse_assignments(raw_generation)
    resolved_assigns = resolve_assigns_for_template(code_str, raw_assigns)
    final_directives = render_directives(code_str, resolved_assigns)

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_directives + "\n")

    payload = {
        "model_name": MODEL_NAME,
        "lora_adapter_dir": LORA_ADAPTER_DIR,
        "tokenizer_dir": TOKENIZER_DIR,
        "code_file": CODE_FILE,
        "prompt_token_count": len(prompt_ids),
        "num_placeholders": num_placeholders,
        "max_new_tokens": max_new_tokens,
        "raw_generation": raw_generation,
        "parsed_assignments": raw_assigns,
        "resolved_assignments": resolved_assigns,
        "final_directives": final_directives,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n================ RAW GENERATION ================\n")
    print(raw_generation)

    print("\n================ FINAL RESOLVED DIRECTIVES ================\n")
    print(final_directives)

    print(f"\nSaved text directives to: {OUT_TXT}")
    print(f"Saved debug JSON to:      {OUT_JSON}")


if __name__ == "__main__":
    run_histogram_inference()
