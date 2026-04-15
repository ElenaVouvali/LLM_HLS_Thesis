import argparse
import json
import math
import os
import random
import re
import gc
import shutil
import numpy as np

from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


# ==============================
# Prompt
# ==============================
PROMPT_TEMPLATE = """
### Role: Expert FPGA/HLS engineer.

### Task:
The kernel marks each directive site with a source marker <SRC_Lk>.
Predict only the directive RHS values for the given optimization goal.
Anchors and directive names are fixed by the source code.

### Kernel
{code}

### Objective
{obj_token}

### Directives
""".lstrip()


def build_prompt(code: str, obj_mode: str) -> str:
    return PROMPT_TEMPLATE.format(
        code=replace_source_labels_with_tokens(code),
        obj_token=GOALS[obj_mode]["token"],
    )


# ==============================
# Objective + placeholder tokens
# ==============================
GOALS = {
    "PARETO_LATENCY_EXTREME": {"token": "<OBJ=PARETO_LATENCY_EXTREME>", "tag": "pareto_latency_extreme"},
    "PARETO_KNEE": {"token": "<OBJ=PARETO_KNEE>", "tag": "pareto_knee"},
    "PARETO_AREA_EXTREME": {"token": "<OBJ=PARETO_AREA_EXTREME>", "tag": "pareto_area_extreme"},
}
GOAL_ORDER = tuple(GOALS.keys())

# Target anchors used in the generated directives
TARGET_PLACEHOLDER_TOKENS = [f"<L{i}>" for i in range(1, 65)]

# Source-only structural markers used inside the kernel code
SOURCE_PLACEHOLDER_TOKENS = [f"<SRC_L{i}>" for i in range(1, 65)]


def source_placeholder_token(label: str) -> str:
    return f"<SRC_{label.upper()}>"


def target_placeholder_token(label: str) -> str:
    return f"<{label.upper()}>"

# ===============================
# Regexes
# ===============================

# source labels like L1: or /* L1: */
SOURCE_LABEL_RE = re.compile(
    r'^\s*(?:/\*\s*(L\d+)\s*:\s*\*/|(L\d+)\s*:)',
    re.IGNORECASE
)

# auto{_PIPE_L1}=..., auto{_UNROLL_L1}=..., auto{_ARRAY_T_L2}=...
TARGET_LINE_LABEL_RE = re.compile(
    r'auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}\s*=',
    re.IGNORECASE
)

ANCHOR_OR_ASSIGN_RE = re.compile(
    r'^\s*(<L\d+>|auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\}\s*=\s*.+)$',
    re.IGNORECASE | re.MULTILINE
)

ASSIGN_RE = re.compile(
    r"^(auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\})\s*=\s*(.+)$",
    re.IGNORECASE,
)

SOURCE_PLACEHOLDER_IN_CODE_RE = re.compile(
    r'auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}',
    re.IGNORECASE
)

LHS_KIND_RE = re.compile(
    r"^auto\{_([A-Z0-9]+(?:_[A-Z0-9]+)*)_L\d+\}$",
    re.IGNORECASE,
)


# ===========================================
# Formatting Helpers (Target Construction)
# ===========================================
def replace_source_labels_with_tokens(text: str) -> str:
    """
    Replace source labels:
        L1: for (...)
        /* L2: */ for (...)
    with source-only structural tokens:
        <SRC_L1> for (...)
        <SRC_L2> for (...)
    """
    if not isinstance(text, str):
        return text

    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        m = SOURCE_LABEL_RE.match(stripped)
        if not m:
            out.append(line)
            continue

        label = (m.group(1) or m.group(2)).upper()
        rest = stripped[m.end():].lstrip()

        src_tok = source_placeholder_token(label)
        if rest:
            out.append(f"{indent}{src_tok} {rest}")
        else:
            out.append(f"{indent}{src_tok}")

    return "\n".join(out)


def extract_source_label_order(source_text: str) -> List[str]:
    order = []
    seen = set()

    for line in source_text.splitlines():
        stripped = line.lstrip()
        m = SOURCE_LABEL_RE.match(stripped)
        if not m:
            continue
        label = (m.group(1) or m.group(2)).upper()
        if label not in seen:
            seen.add(label)
            order.append(label)

    return order


def reorder_target_by_source_order(source_text: str, target_text: str) -> str:
    """
    Reorder raw target assignment lines so that label groups follow the order
    of labels in the source code.
    """
    label_order = extract_source_label_order(source_text)

    grouped = defaultdict(list)
    extras = []

    for raw_line in target_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = TARGET_LINE_LABEL_RE.search(line)
        if m is None:
            extras.append(line)
            continue

        label = m.group(1).upper()
        grouped[label].append(line)

    out = []
    emitted = set()

    for label in label_order:
        if label in grouped:
            out.extend(grouped[label])
            emitted.add(label)

    # keep any leftover labels deterministically at the end
    for label in sorted(grouped.keys()):
        if label not in emitted:
            out.extend(grouped[label])

    out.extend(extras)
    return "\n".join(out)


def extract_ordered_lhs_plan(source_text: str) -> List[Tuple[str, str]]:
    """
    Returns a deterministic ordered plan of directive sites from the source code:
        [("L1", "auto{_ARRAY_T_L1}"), ("L1", "auto{_ARRAY_F_L1}"), ...]
    """
    by_label = defaultdict(list)

    for line in source_text.splitlines():
        for m in SOURCE_PLACEHOLDER_IN_CODE_RE.finditer(line):
            lhs = m.group(0)              # e.g. auto{_PIPE_L3}
            label = m.group(1).upper()    # e.g. L3
            if lhs not in by_label[label]:
                by_label[label].append(lhs)

    plan = []
    for label in extract_source_label_order(source_text):
        for lhs in by_label.get(label, []):
            plan.append((label, lhs))

    return plan


def build_rhs_map_from_target(target_text: str) -> Dict[str, str]:
    """
    Gold RHS values
    Parses: auto{_PIPE_L3} = 1  into: {"auto{_PIPE_L3}": "1"}
    """
    rhs_map = {}

    for raw_line in target_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = ASSIGN_RE.match(line)
        if m is None:
            continue

        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        rhs_map[lhs] = rhs

    return rhs_map



@dataclass
class DeterministicRHSPack:
    input_ids: List[int]          # fixed target-side tokens + RHS tokens interleaved
    labels: List[int]             # -100 for fixed tokens, token id for RHS tokens
    token_weights: List[float]    # 0 for fixed tokens, value weight for RHS tokens
    xattn_target_mask: List[int]  # 1 only on RHS tokens



def build_deterministic_rhs_pack(
    source_text: str,
    target_text: str,
    tok,
    value_w: float = 1.0,
    kind_loss_weights: Optional[Dict[str, float]] = None,
) -> DeterministicRHSPack:
    """
    Build the deterministic target sequence for RHS-only training.
    Fixed schema tokens (<Lk> anchors and "lhs =") are kept in the input as context but are not supervised
    Only RHS value tokens receive labels, loss weights and xattn routing marks. 
    kind_loss_weights --> bias learning toward more important directive kinds 
    """
    rhs_map = build_rhs_map_from_target(target_text)
    full_plan = extract_ordered_lhs_plan(source_text)
    plan = [(label, lhs) for (label, lhs) in full_plan if lhs in rhs_map]

    input_ids, labels, token_weights, xattn_target_mask = [], [], [], []

    def add_fixed(text: str):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend([-100] * len(ids))
        token_weights.extend([0.0] * len(ids))
        xattn_target_mask.extend([0] * len(ids))

    def add_rhs(text: str, weight: float):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend(ids)
        token_weights.extend([weight] * len(ids))
        xattn_target_mask.extend([1] * len(ids))

    current_label = None
    kind_loss_weights = kind_loss_weights or {}

    for label, lhs in plan:
        if label != current_label:
            add_fixed(f"{target_placeholder_token(label)}\n")
            current_label = label

        rhs = rhs_map[lhs].strip()
        kind = lhs_kind(lhs)

        weight = value_w
        weight *= kind_loss_weights.get(kind, 1.0)

        add_fixed(f"{lhs} = ")
        add_rhs(rhs + "\n", weight) # add per-token supervision weighting by directive kind 
                                    # (most difficult directive kind --> larger weight)

    # EOS token is supervised, the model is encouraged to terminate correctly
    eos_ids = tok(tok.eos_token, add_special_tokens=False)["input_ids"]
    input_ids.extend(eos_ids)
    labels.extend(eos_ids)
    token_weights.extend([value_w] * len(eos_ids))
    xattn_target_mask.extend([0] * len(eos_ids))

    return DeterministicRHSPack(
        input_ids=input_ids,
        labels=labels,
        token_weights=token_weights,
        xattn_target_mask=xattn_target_mask,
    )




# =============================
# Dataset Loading
# =============================
def normalize_name(s: str) -> str:
    return re.sub(r"[-\s]+", "_", s.strip().lower())


def normalize_kname(s: str) -> str:
    return normalize_name(s).replace("-", "_")


def family_id_from_kernel_name(name: str) -> str:
    s = normalize_kname(name)

    if s.startswith("machsuite_gemm"):
        return "machsuite_gemm"

    if s.startswith("machsuite_"):
        parts = s.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return s

    if s.startswith("spcl_example"):
        return "spcl_example"

    if s.startswith("serrano_"):
        return "serrano_kalman_filter"

    if s.startswith("rodinia_"):
        rest = s[len("rodinia_"):]
        for special in ["cfd_flux", "cfd_step_factor", "lc_gicov", "lc_mgvf"]:
            if rest.startswith(special):
                return f"rodinia_{special}"
        algo = rest.split("_")[0]
        return f"rodinia_{algo}"

    return s


def load_rows(jsonl_path: str) -> List[dict]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            ex = json.loads(line)
            ex["_jsonl_idx"] = idx
            ex["_family"] = family_id_from_kernel_name(ex["kernel_name"])
            rows.append(ex)
    return rows


def split_by_family(rows: List[dict], val_fams: set, test_fams: set):
    train, val, test = [], [], []
    for r in rows:
        fam = r["_family"]
        if fam in test_fams:
            test.append(r)
        elif fam in val_fams:
            val.append(r)
        else:
            train.append(r)
    return train, val, test


def split_rows_random_design(
    rows: List[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by_kernel: bool = True,
):
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and val_ratio + test_ratio < 1.0):
        raise ValueError("Require 0 <= val_ratio, test_ratio < 1 and val_ratio + test_ratio < 1")

    rng = random.Random(seed)

    def split_bucket(bucket: List[dict]):
        bucket = list(bucket)
        rng.shuffle(bucket)
        n = len(bucket)

        if n <= 2:
            return bucket, [], []

        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))

        if val_ratio > 0 and n_val == 0 and n >= 3:
            n_val = 1
        if test_ratio > 0 and n_test == 0 and n >= 4:
            n_test = 1

        while n_val + n_test >= n:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break

        val = bucket[:n_val]
        test = bucket[n_val:n_val + n_test]
        train = bucket[n_val + n_test:]
        return train, val, test

    if not stratify_by_kernel:
        return split_bucket(rows)

    by_kernel = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel_name"]].append(r)

    train, val, test = [], [], []
    for k in sorted(by_kernel.keys()):
        tr, va, te = split_bucket(by_kernel[k])
        train.extend(tr)
        val.extend(va)
        test.extend(te)

    return train, val, test


def save_split_spec(path: str, train_rows: List[dict], val_rows: List[dict], test_rows: List[dict]):
    dump_json(path, {
        "train_jsonl_idx": [int(r["_jsonl_idx"]) for r in train_rows],
        "val_jsonl_idx": [int(r["_jsonl_idx"]) for r in val_rows],
        "test_jsonl_idx": [int(r["_jsonl_idx"]) for r in test_rows],
    })


def load_split_spec(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_split_spec(rows: List[dict], spec: dict):
    idx_to_row = {int(r["_jsonl_idx"]): r for r in rows}

    train_rows = [idx_to_row[i] for i in spec["train_jsonl_idx"] if i in idx_to_row]
    val_rows   = [idx_to_row[i] for i in spec["val_jsonl_idx"] if i in idx_to_row]
    test_rows  = [idx_to_row[i] for i in spec["test_jsonl_idx"] if i in idx_to_row]
    return train_rows, val_rows, test_rows


def dump_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)



# =====================================================
# Goal-aware point selection
# =====================================================
def pareto_nondominated_mask(rows: List[dict]) -> np.ndarray:
    """
    True = row is non-dominated on (latency, area), assuming both are minimized.
    """
    vals = np.array(
        [[float(r["latency"]), float(r["area"])] for r in rows],
        dtype=np.float64,
    )
    keep = np.ones(len(vals), dtype=bool)

    for i in range(len(vals)):
        dominated = (
            (vals[:, 0] <= vals[i, 0]) &
            (vals[:, 1] <= vals[i, 1]) &
            ((vals[:, 0] < vals[i, 0]) | (vals[:, 1] < vals[i, 1]))
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False

    return keep


def _kernel_normalized_qor(rows: List[dict]):
    lat_vals = np.array([float(r["latency"]) for r in rows], dtype=np.float64)
    area_vals = np.array([float(r["area"]) for r in rows], dtype=np.float64)

    lat_vals = np.log2(np.maximum(lat_vals, 1e-12))

    def minmax(x):
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    lat_n = minmax(lat_vals)
    area_n = minmax(area_vals)
    return lat_n, area_n


def pareto_records_for_kernel(items: List[dict]) -> List[dict]:
    lat_n, area_n = _kernel_normalized_qor(items)
    out = []
    for row, ln, an in zip(items, lat_n, area_n):
        out.append({
            "row": row,
            "lat_n": float(ln),
            "area_n": float(an),
        })

    nd_mask = pareto_nondominated_mask(items)
    for rec, keep in zip(out, nd_mask):
        rec["is_pareto"] = bool(keep)

    frontier = [x for x in out if x["is_pareto"]]
    for x in out:
        x["knee_dist"] = float(math.sqrt(x["lat_n"] ** 2 + x["area_n"] ** 2))
        if x["is_pareto"]:
            x["dom_gap"] = 0.0
            x["dom_count"] = 0
            continue

        dom_count = 0
        for y in out:
            if (
                y["lat_n"] <= x["lat_n"]
                and y["area_n"] <= x["area_n"]
                and (y["lat_n"] < x["lat_n"] or y["area_n"] < x["area_n"])
            ):
                dom_count += 1
        x["dom_count"] = int(dom_count)

        gaps = [
            max(0.0, x["lat_n"] - f["lat_n"]) + max(0.0, x["area_n"] - f["area_n"])
            for f in frontier
        ]
        x["dom_gap"] = float(min(gaps)) if gaps else float("inf")

    return out


def goal_distance_to_ideal(lat_n: float, area_n: float, goal_mode: str) -> float:
    if goal_mode == "PARETO_LATENCY_EXTREME":
        return lat_n
    if goal_mode == "PARETO_AREA_EXTREME":
        return area_n
    if goal_mode == "PARETO_KNEE":
        return math.sqrt(lat_n ** 2 + area_n ** 2)  # distance to ideal point (0, 0)
    raise ValueError(f"Unknown goal_mode: {goal_mode}")


def goal_sort_key(rec: dict, goal_mode: str, domination_penalty: float = 0.0):
    lat_n = float(rec["lat_n"])
    area_n = float(rec["area_n"])
    dom_gap = float(rec.get("dom_gap", 0.0))

    primary = goal_distance_to_ideal(lat_n, area_n, goal_mode)
    if domination_penalty > 0.0:
        primary = primary + domination_penalty * dom_gap

    if goal_mode == "PARETO_LATENCY_EXTREME":
        return (primary, area_n)
    if goal_mode == "PARETO_AREA_EXTREME":
        return (primary, lat_n)
    return (primary, abs(lat_n - area_n), lat_n + area_n)


def canonical_completion_key(source_text: str, target_text: str) -> str:
    """
    Deduplicates by directive assignment (no duplicate in our dataset but keep for robustness)
    If 2 rows have the exact same directive completion, only the best-ranked one is kept.
    """
    target_core = reorder_target_by_source_order(source_text, target_text.strip())
    rhs_map = build_rhs_map_from_target(target_core)

    parts = []
    for label, lhs in extract_ordered_lhs_plan(source_text):
        if lhs in rhs_map:
            parts.append(f"{lhs}={rhs_map[lhs].strip()}")
    return "\n".join(parts)


def score_gap_weight(
    score: float,
    best_score: float,
    worst_score: float,
    w_min: float = 0.6,
    power: float = 1.0,
) -> float:
    """
    Map a per-kernel normalized score to a mild sample weight in [w_min, 1.0].

    score:       lower is better
    best_score:  best score among the chosen top_k for this kernel
    worst_score: worst score among the chosen top_k for this kernel

    power:
      1.0  -> linear decay
      >1.0 -> more aggressive emphasis on the best few
      <1.0 -> flatter weights
    """
    w_min = float(max(0.0, min(1.0, w_min)))
    power = float(max(1e-6, power))

    if worst_score <= best_score + 1e-12:
        return 1.0

    gap = (score - best_score) / (worst_score - best_score)
    gap = max(0.0, min(1.0, gap))

    return float(w_min + (1.0 - w_min) * ((1.0 - gap) ** power))


def rank_goal_candidates(
    rows: List[dict],
    goal_mode: str,
    domination_penalty: float,
    max_dominated_gap: float,
) -> List[dict]:
    decorated = pareto_records_for_kernel(rows)
    frontier = sorted(
        [x for x in decorated if x["is_pareto"]],
        key=lambda x: goal_sort_key(x, goal_mode, domination_penalty=0.0),
    )

    preferred = []
    fallback = []
    for x in decorated:
        if x["is_pareto"]:
            continue
        if float(x["dom_gap"]) <= float(max_dominated_gap):
            preferred.append(x)
        else:
            fallback.append(x)

    preferred.sort(key=lambda x: goal_sort_key(x, goal_mode, domination_penalty=domination_penalty))
    fallback.sort(key=lambda x: goal_sort_key(x, goal_mode, domination_penalty=domination_penalty))
    return frontier + preferred + fallback


def build_local_hard_negative_bank(unique_ranked, hard_neg_top_k=6):
    """
    Use nearby ranked alternatives as contrastive candidates.
    """
    best_row = unique_ranked[0]["row"]
    best_target = reorder_target_by_source_order(best_row["input"], best_row["target"].strip())
    best_rhs = build_rhs_map_from_target(best_target)

    bank = defaultdict(set)

    for rec in unique_ranked[1:hard_neg_top_k]:
        row = rec["row"]
        target_core = reorder_target_by_source_order(row["input"], row["target"].strip())
        rhs_map = build_rhs_map_from_target(target_core)

        for lhs, rhs in rhs_map.items():
            lhs = lhs.upper()
            rhs = rhs.strip()
            if best_rhs.get(lhs, None) != rhs:
                bank[lhs].add(rhs)

    return bank


def build_contrastive_sites_from_sample(
    source_text: str,
    target_text: str,
    prompt_ids: List[int],
    tok,
    max_length: int,
    local_hard_negatives: Optional[Dict[str, List[str]]] = None,
    candidate_sites_per_sample: int = 0,
    candidate_negatives_per_site: int = 0,
    kind_priority: Optional[Dict[str, float]] = None,
):
    """
    For selected directive sites, keep:
      - prefix token ids up to 'lhs = '
      - gold RHS token ids
      - local hard-negative RHS token ids
    """
    if candidate_sites_per_sample <= 0 or candidate_negatives_per_site <= 0:
        return []

    rhs_map = build_rhs_map_from_target(target_text)
    full_plan = extract_ordered_lhs_plan(source_text)
    plan = [(label, lhs) for (label, lhs) in full_plan if lhs in rhs_map]

    local_hard_negatives = local_hard_negatives or {}
    kind_priority = kind_priority or {}

    prefix_ids = list(prompt_ids)
    current_label = None
    sites = []

    for label, lhs in plan:
        if label != current_label:
            anchor_ids = tok(
                f"{target_placeholder_token(label)}\n",
                add_special_tokens=False
            )["input_ids"]

            if len(prefix_ids) + len(anchor_ids) > max_length:
                break

            prefix_ids = prefix_ids + anchor_ids
            current_label = label

        fixed_ids = tok(f"{lhs} = ", add_special_tokens=False)["input_ids"]
        gold_rhs = rhs_map[lhs].strip()
        gold_ids = tok(gold_rhs + "\n", add_special_tokens=False)["input_ids"]

        prefix_for_site = prefix_ids + fixed_ids

        if len(prefix_for_site) + len(gold_ids) > max_length:
            break

        neg_texts = []
        for neg in local_hard_negatives.get(lhs.upper(), []):
            neg = neg.strip()
            if neg and neg != gold_rhs and neg not in neg_texts:
                neg_texts.append(neg)
            if len(neg_texts) >= candidate_negatives_per_site:
                break

        if neg_texts:
            sites.append({
                "label": label,
                "lhs": lhs,
                "kind": lhs_kind(lhs),
                "prefix_ids": prefix_for_site,
                "gold_rhs": gold_rhs,
                "gold_ids": gold_ids,
                "negative_rhs": neg_texts,
                "negative_ids": [
                    tok(neg + "\n", add_special_tokens=False)["input_ids"]
                    for neg in neg_texts
                ],
            })

        prefix_ids = prefix_for_site + gold_ids

    sites.sort(
        key=lambda s: (
            -float(kind_priority.get(s["kind"], 1.0)),
            s["label"],
            s["lhs"],
        )
    )

    return sites[:candidate_sites_per_sample]



def select_goal_rows(
    rows: List[dict],
    goal_mode: str,
    top_k: int,
    domination_penalty: float,
    max_dominated_gap: float,
    score_weight_min: float = 0.6,
    score_weight_power: float = 1.0,
):
    by_kernel = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel_name"]].append(r)

    selected = []
    per_kernel = {}

    for kname, items in by_kernel.items():
        ranked = rank_goal_candidates(
            items,
            goal_mode=goal_mode,
            domination_penalty=domination_penalty,
            max_dominated_gap=max_dominated_gap,
        )

        unique_ranked = []
        seen = set()
        for rec in ranked:
            key = canonical_completion_key(rec["row"]["input"], rec["row"]["target"])
            if key in seen:
                continue
            seen.add(key)
            unique_ranked.append(rec)

        ranked = unique_ranked

        for objective_rank, rec in enumerate(ranked):
            rec["objective_rank"] = objective_rank
            rec["score"] = float(goal_sort_key(rec, goal_mode, domination_penalty=0.0)[0])

        if not ranked:
            continue

        local_hard_negatives = build_local_hard_negative_bank(
            [{"row": rec["row"], "score": rec["score"]} for rec in ranked],
            hard_neg_top_k=max(6, top_k),
        )
        local_hard_negatives = {
            lhs: sorted(vals, key=_rhs_sort_key)
            for lhs, vals in local_hard_negatives.items()
        }

        chosen = ranked[: min(top_k, len(ranked))]

        chosen_scores = [float(rec["score"]) for rec in chosen]
        best_score = min(chosen_scores)
        worst_score = max(chosen_scores)

        chosen_weights = []
        for rec in chosen:
            w = score_gap_weight(
                score=float(rec["score"]),
                best_score=best_score,
                worst_score=worst_score,
                w_min=score_weight_min,
                power=score_weight_power,
            )
            chosen_weights.append(float(w))

            r2 = dict(rec["row"])
            r2["obj_mode"] = goal_mode
            r2["_score"] = float(rec["score"])
            r2["_rank_within_kernel"] = int(rec["objective_rank"])
            r2["_sample_weight"] = float(w)
            r2["_local_hard_negatives"] = local_hard_negatives
            selected.append(r2)

        per_kernel[kname] = {
            "family": items[0]["_family"],
            "n_total": len(items),
            "selected": len(chosen),
            "goal_mode": goal_mode,
            "indices": [rec["row"]["_jsonl_idx"] for rec in chosen],
            "ranks": [rec["objective_rank"] for rec in chosen],
            "scores": [float(rec["score"]) for rec in chosen],
            "sample_weights": chosen_weights,
            "best_score": float(best_score),
            "worst_score": float(worst_score),
            "score_weight_min": float(score_weight_min),
            "score_weight_power": float(score_weight_power),
        }

    return selected, per_kernel


# =======================================================
# Candidate bank + validation decoding for best_stage1
# =======================================================
def build_partial_deterministic_target_text(source_text: str, raw_target: str, min_supervised_sites: int = 1) :
    """
    Build deterministic target text using ONLY directive sites that actually
    exist in the target. This mirrors build_deterministic_rhs_pack().
    """
    target_core = reorder_target_by_source_order(source_text, raw_target.strip())
    rhs_map = build_rhs_map_from_target(target_core)

    out = []
    current_label = None
    n_expected = 0
    n_supervised = 0
    missing_lhs = []

    for label, lhs in extract_ordered_lhs_plan(source_text):
        n_expected += 1
        rhs = rhs_map.get(lhs, None)

        if rhs is None:
            missing_lhs.append(lhs)
            continue

        rhs = rhs.strip()
        if rhs == "" or rhs == "?":
            missing_lhs.append(lhs)
            continue

        if label != current_label:
            out.append(target_placeholder_token(label))
            current_label = label

        out.append(f"{lhs} = {rhs}")
        n_supervised += 1

    if n_supervised < min_supervised_sites:
        raise ValueError(
            f"Too few supervised sites: kept={n_supervised}, expected={n_expected}"
        )

    return "\n".join(out).strip(), {
        "n_expected": n_expected,
        "n_supervised": n_supervised,
        "coverage": (n_supervised / n_expected) if n_expected > 0 else 0.0,
        "missing_lhs": missing_lhs,
    }


def parse_assignment_dict(text: str) -> Dict[str, str]:
    out = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = ASSIGN_RE.match(line)
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
    return out


def canonicalize_generation(text: str) -> str:
    return "\n".join(
        m.group(0).strip()
        for m in ANCHOR_OR_ASSIGN_RE.finditer(text)
    ).strip()


def lhs_kind(lhs: str) -> str:
    m = LHS_KIND_RE.match(lhs.strip())
    if m is None:
        raise ValueError(f"Could not parse lhs kind from: {lhs}")
    return m.group(1).upper()


def _rhs_sort_key(rhs: str):
    s = rhs.strip()
    if re.fullmatch(r"-?\d+", s):
        return (0, int(s), s)
    return (1, s.lower(), s)


LEGAL_RHS_BY_KIND = {   # derived from our dataset
    "PIPE": {"0", "1"},
    "UNROLL": {"0", "2", "3", "4", "5", "8", "10", "15", "16", "17", "31", "32", "34", "36", "62", "63", "64"},
    "ARRAY_T": {"block", "cyclic", "complete"},
    "ARRAY_F": {"0", "2", "4", "8", "16", "32", "64", "128"},
    "ARRAY_D": {"1", "2"},
}


def build_rhs_candidate_bank(rows: List[dict]) -> Dict[str, List[str]]:
    by_kind = defaultdict(set)

    for r in rows:
        target_core = reorder_target_by_source_order(r["input"], r["target"].strip())
        rhs_map = build_rhs_map_from_target(target_core)
        for lhs, rhs in rhs_map.items():
            rhs = rhs.strip()
            if rhs and rhs != "?":
                by_kind[lhs_kind(lhs)].add(rhs)

    out = {}
    for kind in sorted(set(by_kind) | set(LEGAL_RHS_BY_KIND)):
        vals = set(LEGAL_RHS_BY_KIND.get(kind, set()))
        vals.update(by_kind.get(kind, set()))
        out[kind] = sorted(vals, key=_rhs_sort_key)
    return out


def get_rhs_candidates_for_lhs(lhs: str, rhs_candidate_bank: Dict[str, List[str]]) -> List[str]:
    kind = lhs_kind(lhs)
    cands = rhs_candidate_bank.get(kind, [])
    if not cands:
        raise KeyError(f"No RHS candidates found for lhs={lhs} kind={kind}")
    return cands


@torch.no_grad()
def append_token_ids(input_ids, attention_mask, new_ids: List[int]):
    device = input_ids.device
    new_tensor = torch.tensor([new_ids], dtype=input_ids.dtype, device=device)
    new_attn = torch.ones((1, len(new_ids)), dtype=attention_mask.dtype, device=device)
    input_ids = torch.cat([input_ids, new_tensor], dim=1)
    attention_mask = torch.cat([attention_mask, new_attn], dim=1)
    return input_ids, attention_mask


@torch.no_grad()
def score_rhs_candidate_suffix(
    *,
    model,
    tok,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    candidate_text: str,
    routing_start_idx: Optional[torch.Tensor] = None,
    use_harp: bool = False,
):
    device = base_input_ids.device
    cand_ids = tok(candidate_text, add_special_tokens=False)["input_ids"]
    if len(cand_ids) == 0:
        raise ValueError(f"Empty candidate_text tokenization: {repr(candidate_text)}")

    full_input_ids, full_attention_mask = append_token_ids(
        base_input_ids, base_attention_mask, cand_ids
    )

    base_len = int(base_input_ids.shape[1])
    cand_len = len(cand_ids)

    model_inputs = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
    }

    if use_harp:
        if routing_start_idx is None:
            raise ValueError("routing_start_idx is required when use_harp=True")

        xmask = torch.zeros(
            (1, full_input_ids.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        # apply xattn only on the candidate RHS suffix
        xmask[:, base_len:] = 1.0

        model_inputs["routing_start_idx"] = routing_start_idx
        model_inputs["xattn_apply_mask"] = xmask

    outputs = model(**model_inputs)

    cand_logits = outputs.logits[:, base_len - 1: base_len - 1 + cand_len, :].float()
    target = torch.tensor(cand_ids, dtype=torch.long, device=device).unsqueeze(0)

    token_logprobs = F.log_softmax(cand_logits, dim=-1)
    token_logprobs = token_logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1).squeeze(0)

    return {
        "sum_logprob": float(token_logprobs.sum().item()),
        "mean_logprob": float(token_logprobs.mean().item()),
    }


@torch.no_grad()
def constrained_decode_rhs_by_candidate_scoring(
    *,
    model,
    tok,
    prompt_ids: List[int],
    source_text: str,
    rhs_candidate_bank: Dict[str, List[str]],
    score_reduction: str = "mean",
    harp_x: Optional[torch.Tensor] = None,
    harp_mask: Optional[torch.Tensor] = None,
    routing_start_idx: Optional[torch.Tensor] = None,
):
    assert score_reduction in {"mean", "sum"}

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    if routing_start_idx is None:
        routing_start_idx = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)

    parts = []
    current_label = None

    harp_enabled = hasattr(model, "condition_harp") and getattr(model, "initialized_harp_flamingo", False)
    use_harp = harp_enabled and (harp_x is not None) and (harp_mask is not None)

    if use_harp:
        model.condition_harp(harp_x.to(device), harp_mask.to(device))

    try:
        for label, lhs in extract_ordered_lhs_plan(source_text):
            if label != current_label:
                anchor_text = f"{target_placeholder_token(label)}\n"
                anchor_ids = tok(anchor_text, add_special_tokens=False)["input_ids"]
                input_ids, attention_mask = append_token_ids(input_ids, attention_mask, anchor_ids)
                parts.append(anchor_text)
                current_label = label

            prefix_text = f"{lhs} = "
            prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
            input_ids, attention_mask = append_token_ids(input_ids, attention_mask, prefix_ids)
            parts.append(prefix_text)

            candidates = get_rhs_candidates_for_lhs(lhs, rhs_candidate_bank)

            scored = []
            for rhs in candidates:
                stats = score_rhs_candidate_suffix(
                    model=model,
                    tok=tok,
                    base_input_ids=input_ids,
                    base_attention_mask=attention_mask,
                    candidate_text=rhs + "\n",
                    routing_start_idx=routing_start_idx,
                    use_harp=use_harp,
                )
                scored.append({
                    "rhs": rhs,
                    "score": stats["mean_logprob"] if score_reduction == "mean" else stats["sum_logprob"],
                    "mean_logprob": stats["mean_logprob"],
                    "sum_logprob": stats["sum_logprob"],
                })

            scored.sort(key=lambda x: (x["score"], x["sum_logprob"]), reverse=True)
            best = scored[0]

            chosen_text = best["rhs"] + "\n"
            chosen_ids = tok(chosen_text, add_special_tokens=False)["input_ids"]
            input_ids, attention_mask = append_token_ids(input_ids, attention_mask, chosen_ids)
            parts.append(chosen_text)

        return "".join(parts).rstrip()

    finally:
        if hasattr(model, "clear_harp"):
            model.clear_harp()



def evaluate_prediction(reference_target: str, raw_generation: str) -> Dict[str, object]:
    pred_text = canonicalize_generation(raw_generation)
    ref_text = reference_target.strip()

    ref_assign = parse_assignment_dict(ref_text)
    pred_assign = parse_assignment_dict(pred_text)

    expected_keys = list(ref_assign.keys())
    exact_value_match_count = sum(
        (k in pred_assign) and (pred_assign[k] == ref_assign[k])
        for k in expected_keys
    )

    return {
        "canonical_prediction": pred_text,
        "value_accuracy_over_expected": exact_value_match_count / max(len(expected_keys), 1),
    }


@dataclass
class SelectionCase:
    kernel_name: str
    obj_mode: str
    source_text: str
    reference_target: str


def build_selection_cases(
    val_rows: List[dict],
    goal_mode: str,
    max_kernels: int = 4,
    min_coverage: float = 0.85,
    min_supervised_sites: int = 4,
) -> List[SelectionCase]:
    by_kernel = defaultdict(list)
    for r in val_rows:
        if r["obj_mode"] == goal_mode:
            by_kernel[r["kernel_name"]].append(r)

    cases = []
    kernels_kept = 0

    for kernel_name in sorted(by_kernel.keys()):
        items = by_kernel[kernel_name]
        best = sorted(
            items,
            key=lambda r: (
                int(r.get("_rank_within_kernel", 10**9)),
                float(r.get("_score", 10**9)),
            ),
        )[0]

        try:
            ref_target, ref_meta = build_partial_deterministic_target_text(
                best["input"],
                best["target"],
                min_supervised_sites=min_supervised_sites,
            )
        except ValueError:
            continue

        if ref_meta["coverage"] < min_coverage:
            continue

        cases.append(
            SelectionCase(
                kernel_name=kernel_name,
                obj_mode=goal_mode,
                source_text=best["input"],
                reference_target=ref_target,
            )
        )
        kernels_kept += 1

        if kernels_kept >= max_kernels:
            break

    return cases


class StageValSelectionCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        selection_cases: List[SelectionCase],
        rhs_candidate_bank: Dict[str, List[str]],
        output_dir: str,
        max_prompt_tokens: int = 7168,
        candidate_score_reduction: str = "mean",
        best_dir_name: str = "best_custom_stage1",
        mem_bank: Optional[Dict[str, dict]] = None,
        mem_dim: int = 32,
        max_slots: int = 64,
    ):
        self.tok = tokenizer
        self.selection_cases = selection_cases
        self.rhs_candidate_bank = rhs_candidate_bank
        self.output_dir = output_dir
        self.max_prompt_tokens = max_prompt_tokens
        self.candidate_score_reduction = candidate_score_reduction
        self.best_dir_name = best_dir_name
        self.mem_bank = mem_bank or {}
        self.mem_dim = mem_dim
        self.max_slots = max_slots
        self.best_score = -1e18
        self.best_step = -1

    def _run_case(self, model, case: SelectionCase) -> dict:
        prompt = build_prompt(case.source_text, case.obj_mode)

        enc = self.tok(prompt, add_special_tokens=False)
        prompt_ids = enc["input_ids"][-self.max_prompt_tokens:] if len(enc["input_ids"]) > self.max_prompt_tokens else enc["input_ids"]

        device = next(model.parameters()).device
        routing_start_idx = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)

        harp_x = None
        harp_mask = None
        if hasattr(model, "initialized_harp_flamingo") and getattr(model, "initialized_harp_flamingo", False):
            harp_x, harp_mask = get_real_memory_pack_for_kernel(
                self.mem_bank,
                case.kernel_name,
                self.max_slots,
                self.mem_dim,
            )

        pred = constrained_decode_rhs_by_candidate_scoring(
            model=model,
            tok=self.tok,
            prompt_ids=prompt_ids,
            source_text=case.source_text,
            rhs_candidate_bank=self.rhs_candidate_bank,
            score_reduction=self.candidate_score_reduction,
            harp_x=harp_x,
            harp_mask=harp_mask,
            routing_start_idx=routing_start_idx,
        )

        metrics = evaluate_prediction(case.reference_target, pred)
        return {
            "kernel_name": case.kernel_name,
            "obj_mode": case.obj_mode,
            "reference_target": case.reference_target,
            "prediction": metrics["canonical_prediction"],
            "value_accuracy_over_expected": float(metrics["value_accuracy_over_expected"]),
        }

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        model = kwargs["model"]
        was_training = model.training
        model.eval()

        try:
            rows = [self._run_case(model, case) for case in self.selection_cases]
            mean_value_acc = float(sum(r["value_accuracy_over_expected"] for r in rows) / max(len(rows), 1))
            selection_score = mean_value_acc

            print("\n" + "=" * 100)
            print(f"[VAL-SELECTION] step={state.global_step}")
            print(f"[VAL-SELECTION] mean_value_acc={mean_value_acc:.6f}")
            print(f"[VAL-SELECTION] selection_score={selection_score:.6f}")
            print("=" * 100)

            metrics_obj = {
                "step": int(state.global_step),
                "mean_value_acc": mean_value_acc,
                "selection_score": selection_score,
                "rows": rows,
            }

            dump_json(
                os.path.join(self.output_dir, f"val_selection_step_{state.global_step}.json"),
                metrics_obj,
            )

            if selection_score > self.best_score:
                self.best_score = selection_score
                self.best_step = int(state.global_step)

                best_dir = os.path.join(self.output_dir, self.best_dir_name)
                if os.path.isdir(best_dir):
                    shutil.rmtree(best_dir)

                model.save_pretrained(best_dir)
                self.tok.save_pretrained(best_dir)

                if hasattr(model, "initialized_harp_flamingo") and getattr(model, "initialized_harp_flamingo", False):
                    harp_sd = get_harp_xattn_state_dict(model)
                    if harp_sd:
                        torch.save(harp_sd, os.path.join(best_dir, "harp_xattn.pt"))
                        print(f"[VAL-SELECTION] Saved best HARP xattn weights -> {os.path.join(best_dir, 'harp_xattn.pt')}")

                dump_json(
                    os.path.join(best_dir, "best_selection_metrics.json"),
                    metrics_obj,
                )

                print(f"[VAL-SELECTION] New best checkpoint at step {state.global_step} -> {best_dir}")

        finally:
            if was_training:
                model.train()


# ==========================================
# HARP memory bank loader (.memory.pt files)
# ==========================================
def load_memory_bank(memory_dir: str) -> Dict[str, dict]:
    """
    Expects each *.memory.pt to contain:
      - node_embs: [max_slots, mem_dim] (float)
      - node_embs_mask: [max_slots] (bool)
      - labels: [mem_dim] (int)
    """
    bank = {}
    for fn in os.listdir(memory_dir):
        if not fn.endswith(".memory.pt"):
            continue

        pack = torch.load(os.path.join(memory_dir, fn), map_location="cpu", weights_only=False)
        k = fn.replace(".memory.pt", "")

        kv = pack["node_embs"].float()
        mask = pack["node_embs_mask"].bool()
        labels = pack.get("labels", None)

        if labels is not None:
            active = []
            for i, (lbl, m) in enumerate(zip(labels, mask.tolist())):
                lbl = int(lbl)
                if m and lbl > 0:
                    active.append((lbl, kv[i]))

            active.sort(key=lambda x: x[0])

            dense_kv = torch.zeros_like(kv)
            dense_mask = torch.zeros_like(mask)

            for j, (lbl, vec) in enumerate(active):
                assert lbl == j + 1, f"{fn}: non-contiguous labels { [x[0] for x in active] }"
                dense_kv[j] = vec
                dense_mask[j] = True

            kv = dense_kv.contiguous()
            mask = dense_mask.contiguous()

        bank[k] = {"kv": kv.contiguous(), "mask": mask.contiguous()}
        bank[normalize_kname(k)] = bank[k]

    return bank




# ================================
# HARP / Flamingo-style utilities
# ================================
def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation."""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (mixin, base_cls), {})


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)





def print_xattn_gate_stats(model, print_grads=True):
    attn_gates = []
    ff_gates = []

    for n, p in model.named_parameters():
        if n.endswith("attn_gate") or n.endswith("ff_gate"):
            raw = float(p.detach().cpu().item())
            tanh_val = float(p.detach().cpu().tanh().item())

            grad = None
            grad_abs = None
            if print_grads and p.grad is not None:
                grad = float(p.grad.detach().cpu().item())
                grad_abs = abs(grad)

            row = {
                "name": n,
                "raw": raw,
                "tanh": tanh_val,
                "grad": grad,
                "grad_abs": grad_abs,
            }

            if n.endswith("attn_gate"):
                attn_gates.append(row)
            else:
                ff_gates.append(row)

    print("[GATES] attn gates:")
    for row in attn_gates:
        if row["grad"] is None:
            print(
                f"  {row['name']}: "
                f"raw={row['raw']:.8f} tanh={row['tanh']:.8f} grad=None"
            )
        else:
            print(
                f"  {row['name']}: "
                f"raw={row['raw']:.8f} tanh={row['tanh']:.8f} "
                f"grad={row['grad']:.8e} |grad|={row['grad_abs']:.8e}"
            )

    print("[GATES] ff gates:")
    for row in ff_gates:
        if row["grad"] is None:
            print(
                f"  {row['name']}: "
                f"raw={row['raw']:.8f} tanh={row['tanh']:.8f} grad=None"
            )
        else:
            print(
                f"  {row['name']}: "
                f"raw={row['raw']:.8f} tanh={row['tanh']:.8f} "
                f"grad={row['grad']:.8e} |grad|={row['grad_abs']:.8e}"
            )


def print_xattn_forward_stats(model):
    found = False
    for name, module in model.named_modules():
        if isinstance(module, MaskedCrossAttention) and getattr(module, "last_debug", None):
            found = True
            dbg = module.last_debug
            print(f"[XATTN-DBG] {name}: {dbg}")
    if not found:
        print("[XATTN-DBG] no cross-attn forward stats collected yet")


def get_harp_xattn_state_dict(model):
    sd = model.state_dict()
    return {
        k: v.detach().cpu()
        for k, v in sd.items()
        if "gated_cross_attn_layer" in k
    }


def get_first_real_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0")


def move_harp_modules_to_model_device(model):
    device = get_first_real_device(model)
    moved = 0
    for module in model.modules():
        if isinstance(module, GatedCrossAttentionBlock):
            module.to(device=device)
            moved += 1
    print(f"[HARP-DEVICE] moved {moved} HARP blocks to {device}")


def load_partial_harp_xattn(model, harp_xattn_path: str, tag: str):
    if not harp_xattn_path or not os.path.isfile(harp_xattn_path):
        print(f"[{tag}] no harp_xattn.pt found at: {harp_xattn_path}")
        return

    harp_sd = torch.load(harp_xattn_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(harp_sd, strict=False)

    harp_missing = [k for k in missing if "gated_cross_attn_layer" in k]
    print(f"[{tag}] harp_missing[:10]={harp_missing[:10]}")
    print(f"[{tag}] unexpected[:10]={unexpected[:10]}")

    move_harp_modules_to_model_device(model)


def infer_decoder_layers_attr_name(model) -> str:
    candidates = [
        "base_model.model.model.layers",
        "base_model.model.decoder.layers",
        "base_model.model.transformer.h",
        "base_model.model.gpt_neox.layers",
        "model.layers",
        "decoder.layers",
        "transformer.h",
    ]
    for att in candidates:
        try:
            val = getattr_recursive(model, att)
            if isinstance(val, (nn.ModuleList, list)) and len(val) > 0:
                return att
        except Exception:
            continue
    raise ValueError(
        "Could not infer decoder layer path. Please add the correct recursive path for this backbone."
    )


def get_real_memory_pack_for_kernel(
    mem_bank: Dict[str, dict],
    kernel_name: str,
    max_slots: int,
    mem_dim: int,
):
    pack = mem_bank.get(kernel_name) or mem_bank.get(normalize_kname(kernel_name))
    if pack is None:
        return (
            torch.zeros((1, max_slots, mem_dim), dtype=torch.float32),
            torch.zeros((1, max_slots), dtype=torch.bool),
        )
    return (
        pack["kv"].unsqueeze(0).float(),
        pack["mask"].unsqueeze(0).bool(),
    )



# ==============================
# Cross-Attention Modules
# ==============================
def exists(val):
    return val is not None


def FeedForward(dim: int, mult: int = 4) -> nn.Module:
    inner = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner, bias=False),
        nn.GELU(),
        nn.Linear(inner, dim, bias=False),
    )


class MaskedCrossAttention(nn.Module):
    """
    Attention from LM hidden states to HARP memory slots.

    Routing uses ONLY target placeholder anchors <Lk>.
    Source structural markers <SRC_Lk> are visible to the LM, but they do not
    participate in cross-attention slot routing.
    """
    def __init__(
        self,
        *,
        dim,
        dim_memory,
        dim_head=64,
        heads=8,
        only_attend_immediate_memory=True,
        mask_mode="segment",   # "segment" or "token"
    ):
        super().__init__()
        assert mask_mode in {"segment", "token"}

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.mask_mode = mask_mode
        self.only_attend_immediate_memory = only_attend_immediate_memory    # for "segment" mode --> a token can see only 
                                                                            # the most resent placeholder's slot or all the previous ones too

        self.last_debug = {}

        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_memory, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(
        self,
        x,
        memory,
        placeholder_slot_ids=None,
        memory_mask=None,
        use_cached_memory=False,
    ):

        """
        x (text):              [B, T_txt, D_lm]
        memory (graph):        [B, S, D_mem]
        placeholder_locations: [B, T_txt] bool
        memory_mask:           [B, S] bool
        """

        B, T_txt, _ = x.shape
        _, S, _ = memory.shape
        h = self.heads

        if not use_cached_memory:
            assert exists(placeholder_slot_ids), "placeholder_slot_ids is required unless use_cached_memory=True"

        x = self.norm(x)
        memory = memory.to(dtype=x.dtype)

        q = self.to_q(x)
        k, v = self.to_kv(memory).chunk(2, dim=-1)

        q, k, v = rearrange_many(
            (q, k, v),
            "b n (h d) -> b h n d",
            h=h,
        )

        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        memory_slots = torch.arange(1, S + 1, device=x.device, dtype=torch.long)  # [S]

        if exists(placeholder_slot_ids):
            if use_cached_memory:
                active_slot_ids = last_seen_slot_id(placeholder_slot_ids).expand(B, T_txt)
            else:
                if self.mask_mode == "segment":
                    active_slot_ids = forward_fill_slot_ids(placeholder_slot_ids)
                elif self.mask_mode == "token":
                    active_slot_ids = placeholder_slot_ids
                else:
                    raise NotImplementedError()

            text_to_memory_mask = torch.eq(
                rearrange(active_slot_ids, "b t -> b 1 t 1"),
                rearrange(memory_slots, "s -> 1 1 1 s"),
            )

            if self.mask_mode == "token" and not use_cached_memory:
                text_to_memory_mask = text_to_memory_mask & rearrange(
                    placeholder_slot_ids.ne(0), "b t -> b 1 t 1"
                )

            if self.mask_mode == "segment" and not self.only_attend_immediate_memory:
                text_to_memory_mask = torch.ge(
                    rearrange(active_slot_ids, "b t -> b 1 t 1"),
                    rearrange(memory_slots, "s -> 1 1 1 s"),
                )

            if exists(memory_mask):
                text_to_memory_mask = text_to_memory_mask & rearrange(
                    memory_mask, "b s -> b 1 1 s"
                )

            sim = sim.masked_fill(~text_to_memory_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(placeholder_slot_ids):
            text_without_memory_mask = ~text_to_memory_mask.any(dim=-1, keepdim=True)
            attn = attn.masked_fill(text_without_memory_mask, 0.0)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        with torch.no_grad():
            dbg = {
                "B": int(B),
                "T_txt": int(T_txt),
                "S": int(S),
                "memory_mask_true": int(memory_mask.sum().item()) if exists(memory_mask) else None,
                "out_abs_mean": float(out.abs().mean().item()),
                "out_l2_mean": float(out.float().norm(dim=-1).mean().item()),
            }

            if exists(placeholder_slot_ids):
                dbg["placeholder_tokens"] = int(placeholder_slot_ids.ne(0).sum().item())
                dbg["active_tokens_after_fill"] = int(active_slot_ids.ne(0).sum().item())

                if 'text_to_memory_mask' in locals():
                    dbg["valid_edges"] = int(text_to_memory_mask.sum().item())
                    dbg["tokens_with_route"] = int(text_to_memory_mask.any(dim=-1).sum().item())

            dbg["attn_mean"] = float(attn.mean().item())
            dbg["attn_max"] = float(attn.max().item())
            self.last_debug = dbg

        return self.to_out(out)



class GatedCrossAttentionBlock(nn.Module):
    """
    Flamingo-style gated xattn + FF block
    """    
    def __init__(self, *, dim, dim_memory, dim_head=64, heads=8,
                 ff_mult=4, only_attend_immediate_memory=True,
                 mask_mode="segment", enable_ff=True,
                 attn_gate_init=0.05, ff_gate_init=0.05):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_memory=dim_memory,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_memory=only_attend_immediate_memory,
            mask_mode=mask_mode,
        )
        self.attn_gate = nn.Parameter(torch.tensor([attn_gate_init]))
        self.enable_ff = enable_ff

        if enable_ff:
            self.ff = FeedForward(dim, mult=ff_mult)
            self.ff_gate = nn.Parameter(torch.tensor([ff_gate_init]))
        else:
            self.ff = None
            self.register_parameter("ff_gate", None)

    def forward(self, x, memory, placeholder_slot_ids=None, memory_mask=None, use_cached_memory=False, xattn_apply_mask=None):
        attn_out = self.attn(
            x,
            memory,
            placeholder_slot_ids=placeholder_slot_ids,
            memory_mask=memory_mask,
            use_cached_memory=use_cached_memory,
        )

        if xattn_apply_mask is not None:
            mask = xattn_apply_mask.to(device=attn_out.device, dtype=attn_out.dtype)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)   # [B, T, 1]
            attn_out = attn_out * mask

        x = x + attn_out * self.attn_gate.tanh()

        if self.ff is not None:
            ff_out = self.ff(x)
            if xattn_apply_mask is not None:
                ff_out = ff_out * mask
            x = x + ff_out * self.ff_gate.tanh()

        return x



def build_placeholder_slot_ids(input_ids, placeholder_token_ids, routing_start_idx=None):
    """
    Returns:
        slot_ids [B, T] long
            0 = not a placeholder token
            1 = <L1>
            2 = <L2>
            ...
    """
    slot_ids = torch.zeros_like(input_ids, dtype=torch.long)
    for slot_idx, tok_id in enumerate(placeholder_token_ids, start=1):
        slot_ids[input_ids == tok_id] = slot_idx

    if routing_start_idx is not None:
        if routing_start_idx.ndim == 0:
            routing_start_idx = routing_start_idx.unsqueeze(0)
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        valid = pos >= routing_start_idx.unsqueeze(1)   # only target anchors are allowed to activate HARP memory routing
        slot_ids = torch.where(valid, slot_ids, torch.zeros_like(slot_ids))

    return slot_ids



def forward_fill_slot_ids(slot_ids):
    """
    slot_ids: [B, T], values in {0,1,...,S}
    Segment Mode : Returns active slot id at each position, by carrying forward the most recent nonzero slot.
    """
    B, T = slot_ids.shape
    pos = torch.arange(T, device=slot_ids.device).unsqueeze(0).expand(B, T)
    seen_pos = torch.where(slot_ids.ne(0), pos, torch.full_like(pos, -1))
    last_pos = torch.cummax(seen_pos, dim=1).values
    gather_pos = last_pos.clamp(min=0)
    active = slot_ids.gather(1, gather_pos)
    active = torch.where(last_pos.ge(0), active, torch.zeros_like(active))
    return active


def last_seen_slot_id(slot_ids):
    """
    slot_ids: [B, T_prev]
    Returns:
        [B, 1] = last nonzero slot id seen in the cached prefix --> Cached Generation
    """
    B, T = slot_ids.shape
    pos = torch.arange(T, device=slot_ids.device).unsqueeze(0).expand(B, T)
    seen_pos = torch.where(slot_ids.ne(0), pos, torch.full_like(pos, -1))
    last_pos = seen_pos.max(dim=1).values
    gather_pos = last_pos.clamp(min=0).unsqueeze(1)
    last_slot = slot_ids.gather(1, gather_pos)
    last_slot = torch.where(last_pos.ge(0).unsqueeze(1), last_slot, torch.zeros_like(last_slot))
    return last_slot


class HARPLayer(nn.Module):
    """
    Thin wrapper around a decoder layer that optionally inserts a gated HARP cross-attention block before the decoder layer.
    """
    def __init__(self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.harp_x = None
        self.harp_mask = None
        self.placeholder_slot_ids = None
        self.use_cached_memory = False
        self.xattn_apply_mask = None

        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = gradient_checkpointing
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def condition_xattn_apply_mask(self, xattn_apply_mask):
        self.xattn_apply_mask = xattn_apply_mask

    def is_conditioned(self) -> bool:
        return self.harp_x is not None and self.harp_mask is not None and self.placeholder_slot_ids is not None

    def condition_harp_x(self, harp_x, harp_mask):
        self.harp_x = harp_x
        self.harp_mask = harp_mask

    def condition_placeholder_slot_ids(self, placeholder_slot_ids):
        self.placeholder_slot_ids = placeholder_slot_ids

    def condition_use_cached_memory(self, use_cached_memory):
        self.use_cached_memory = use_cached_memory

    def forward(self, *args, **kwargs):
        if args:
            hidden_states = args[0]
            rest_args = args[1:]
        else:
            if "hidden_states" not in kwargs:
                raise ValueError("HARPLayer.forward requires hidden_states")
            hidden_states = kwargs["hidden_states"]
            rest_args = ()

        if self.gated_cross_attn_layer is not None:
            if self.harp_x is None or self.harp_mask is None:
                raise ValueError("HARP memory must be conditioned before forward pass")
            if self.placeholder_slot_ids is None:
                raise ValueError("placeholder_slot_ids must be conditioned before forward pass")

            hidden_states = self.gated_cross_attn_layer(
                hidden_states,
                self.harp_x,
                placeholder_slot_ids=self.placeholder_slot_ids,
                memory_mask=self.harp_mask,
                use_cached_memory=bool(self.use_cached_memory),
                xattn_apply_mask=self.xattn_apply_mask,
            )

        if args:
            return self.decoder_layer(hidden_states, *rest_args, **kwargs)
        else:
            kwargs["hidden_states"] = hidden_states
            return self.decoder_layer(**kwargs)
    


class HARPLMMixin(nn.Module):
    """
    - Wraps decoder layers with HARPLayer
    - Builds placeholder_slot_ids from the input ids (in which slot we are)
    - Conditioning (gives harp_x, harp_mask)
    - Keeps cached memory during generation
    """
    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_harp_flamingo(
        self,
        placeholder_token_ids,
        lang_hidden_size,
        mem_hidden_size,
        cross_attn_every_n_layers,
        gradient_checkpointing,
        xattn_heads=8,
        xattn_dim_head=64,
        xattn_ff_mult=4,
        only_attend_immediate_memory=True,
        mask_mode="segment",
    ):
        self.old_decoder_blocks = self._get_decoder_layers()
        wrapped_layers = []
        for layer_idx, decoder_layer in enumerate(self.old_decoder_blocks):
            gated_cross_attn_layer = None
            if (layer_idx + 1) % cross_attn_every_n_layers == 0:
                gated_cross_attn_layer = GatedCrossAttentionBlock(
                    dim=lang_hidden_size,
                    dim_memory=mem_hidden_size,
                    dim_head=xattn_dim_head,
                    heads=xattn_heads,
                    ff_mult=xattn_ff_mult,
                    only_attend_immediate_memory=only_attend_immediate_memory,
                    mask_mode=mask_mode,
                    enable_ff=True,
                    attn_gate_init=0.05,
                    ff_gate_init=0.05,
                )

            wrapped_layers.append(
                HARPLayer(
                    gated_cross_attn_layer=gated_cross_attn_layer,
                    decoder_layer=decoder_layer,
                    gradient_checkpointing=gradient_checkpointing,
                )
            )

        self._set_decoder_layers(nn.ModuleList(wrapped_layers))
        self.placeholder_token_ids = tuple(int(x) for x in placeholder_token_ids)
        self.initialized_harp_flamingo = True
        self._use_cached_harp_x = False

    def condition_harp(self, harp_x, harp_mask):
        for layer in self._get_decoder_layers():
            if isinstance(layer, HARPLayer):
                layer.condition_harp_x(harp_x, harp_mask)
        # Keep cache enabled so HF generate() can reuse the same memory across
        # repeated forward() calls after the prompt step.
        self._use_cached_harp_x = True

    def clear_harp(self):
        for layer in self._get_decoder_layers():
            if isinstance(layer, HARPLayer):
                layer.condition_harp_x(None, None)
                layer.condition_placeholder_slot_ids(None)
                layer.condition_use_cached_memory(False)
                layer.condition_xattn_apply_mask(None)
        self._use_cached_harp_x = False

    def is_conditioned(self) -> bool:
        return all(
            (not isinstance(layer, HARPLayer)) or layer.is_conditioned()
            for layer in self._get_decoder_layers()
        )
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        routing_start_idx=None,
        xattn_apply_mask=None,
        **kwargs,
    ):
        if not getattr(self, "initialized_harp_flamingo", False):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        if input_ids is None:
            raise ValueError("input_ids must be provided for HARP-Flamingo forward")

        placeholder_slot_ids = build_placeholder_slot_ids(
            input_ids,
            self.placeholder_token_ids,
            routing_start_idx=routing_start_idx,
        )

        use_cached_placeholder_locations = (
            self._use_cached_harp_x
            and self.is_conditioned()
            and not placeholder_slot_ids.ne(0).any()
        )

        for layer in self._get_decoder_layers():
            if not isinstance(layer, HARPLayer):
                continue
            if not use_cached_placeholder_locations:
                layer.condition_placeholder_slot_ids(placeholder_slot_ids)
            layer.condition_use_cached_memory(use_cached_placeholder_locations)
            layer.condition_xattn_apply_mask(xattn_apply_mask)

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        if labels is not None:
            kwargs["labels"] = labels

        return super().forward(**kwargs)


class SaveHarpXattnCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        harp_sd = get_harp_xattn_state_dict(model)
        if harp_sd:
            torch.save(harp_sd, os.path.join(ckpt_dir, "harp_xattn.pt"))
            print(f"[HARP-SAVE] saved xattn weights to {ckpt_dir}/harp_xattn.pt")
        return control





# ====================================
# Dataset + Pad Collator (SFT)
# ====================================
class SFTDataset(Dataset):
    """
    Constructs prompt / target samples and optional per-site contrastive metadata.
    """
    def __init__(
        self,
        rows: List[dict],
        tok,
        max_length: int,
        value_loss_weight: float = 1.0,
        candidate_sites_per_sample: int = 0,
        candidate_negatives_per_site: int = 0,
    ):
        self.samples = []
        self.lengths = []
        self.tok = tok
        self.max_length = max_length

        n_total = 0
        n_missing_obj = 0

        kind_loss_weights = {
            "UNROLL": 1.6,
            "ARRAY_F": 1.2,
            "PIPE": 1.2,
            "ARRAY_T": 1.0,
            "ARRAY_D": 0.8,
        }

        for ex in rows:
            obj_token = GOALS[ex["obj_mode"]]["token"]
            prompt = build_prompt(ex["input"], ex["obj_mode"])
            target_core = reorder_target_by_source_order(ex["input"], ex["target"].strip())

            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]

            det_pack = build_deterministic_rhs_pack(
                ex["input"],
                target_core,
                tok,
                value_w=value_loss_weight,
                kind_loss_weights=kind_loss_weights,
            )

            t_ids = det_pack.input_ids
            target_labels = det_pack.labels
            plan_loss_weights = det_pack.token_weights
            plan_xattn_target_mask = det_pack.xattn_target_mask

            if len(t_ids) >= max_length:
                t_ids = t_ids[:max_length]
                target_labels = target_labels[:max_length]
                plan_loss_weights = plan_loss_weights[:max_length]
                plan_xattn_target_mask = plan_xattn_target_mask[:max_length]
                p_ids = []
            else:
                max_p = max_length - len(t_ids)
                if len(p_ids) > max_p:
                    p_ids = p_ids[-max_p:]

            obj_id = tok.encode(obj_token, add_special_tokens=False)[0]
            seen_obj = (obj_id in p_ids)    # verify if the objective token has survived
            n_total += 1
            if not seen_obj:
                n_missing_obj += 1

            input_ids = p_ids + t_ids
            attn = [1] * len(input_ids)

            labels = [-100] * len(p_ids) + target_labels
            token_weights = [0.0] * len(p_ids) + plan_loss_weights

            full_xattn_target_mask = [0] * len(p_ids) + plan_xattn_target_mask
            xattn_apply_mask = full_xattn_target_mask[1:] + [0]

            contrastive_sites = build_contrastive_sites_from_sample(
                source_text=ex["input"],
                target_text=target_core,
                prompt_ids=p_ids,
                tok=tok,
                max_length=max_length,
                local_hard_negatives=ex.get("_local_hard_negatives", {}),
                candidate_sites_per_sample=candidate_sites_per_sample,
                candidate_negatives_per_site=candidate_negatives_per_site,
                kind_priority=kind_loss_weights,
            )

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "token_weights": torch.tensor(token_weights, dtype=torch.float32),
                "xattn_apply_mask": torch.tensor(xattn_apply_mask, dtype=torch.float32),
                "sample_weight": torch.tensor(float(ex.get("_sample_weight", 1.0)), dtype=torch.float32),
                "kernel_name": ex["kernel_name"],
                "routing_start_idx": torch.tensor(len(p_ids), dtype=torch.long),
                "contrastive_sites": contrastive_sites,
            })

            self.lengths.append(len(input_ids))

        if n_total > 0:
            missing_pct = (n_missing_obj / n_total) * 100
            print("\n--- Truncation Summary ---")
            print(f"Total samples: {n_total}")
            print(f"Missing objective token: {n_missing_obj} ({missing_pct:.2f}%)")
            if missing_pct > 20:
                print("WARNING: High truncation rate detected. Consider increasing max_length.")
            print("--------------------------\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PadCollator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)

        def pad_1d(t, pad_value):
            if t.shape[0] == max_len:
                return t
            pad = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)

        input_ids = torch.stack([pad_1d(x["input_ids"], self.tok.pad_token_id) for x in batch])
        attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in batch])
        labels = torch.stack([pad_1d(x["labels"], -100) for x in batch])
        sample_weight = torch.stack([x["sample_weight"] for x in batch])
        kernel_name = [x["kernel_name"] for x in batch]
        routing_start_idx = torch.stack([x["routing_start_idx"] for x in batch])
        token_weights = torch.stack([pad_1d(x["token_weights"], 0.0) for x in batch])
        xattn_apply_mask = torch.stack([pad_1d(x["xattn_apply_mask"], 0.0) for x in batch])
        contrastive_sites = [x["contrastive_sites"] for x in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_weight": sample_weight,
            "kernel_name": kernel_name,
            "routing_start_idx": routing_start_idx,
            "token_weights": token_weights,
            "xattn_apply_mask": xattn_apply_mask,
            "contrastive_sites": contrastive_sites,
        }



# ========================================
# Model Helpers for Stage_1 (simple SFT)
# ========================================
def get_input_embeddings_module(model):
    emb = None

    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()

    if emb is None and hasattr(model, "base_model") and hasattr(model.base_model, "get_input_embeddings"):
        emb = model.base_model.get_input_embeddings()

    if emb is None:
        raise ValueError("Could not access model input embeddings.")

    return emb


def unfreeze_input_embeddings(model):
    """
    Newly added special tokens need trainable input embeddings.
    """
    emb = get_input_embeddings_module(model)
    emb.weight.requires_grad_(True)

    # Helpful for some HF/PEFT stacks
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return emb




# =========================================
# Trainer (SFT : Stage_1 + Stage_2)
# =========================================
class LengthGroupedTrainer(Trainer):
    """
    - Length-grouped sampling + Per-sample weights
    - Conditions HARP memory per batch using kernel_name
    - Computes chunked CE to avoid giant [B*T,V] flatten allocations
    """
    def __init__(
        self,
        *args,
        group_by_length: bool = False,
        mem_bank: Optional[Dict[str, dict]] = None,
        mem_dim: int = 32,
        max_slots: int = 64,
        lr_lora: float = 2e-4,
        lr_xattn: float = 1e-4,
        lr_gate: float = 1e-3,
        lr_ff: float = 0.0,
        lr_gate_ff: float = 0.0,
        lr_embed: Optional[float] = None,
        loss_chunk_t: int = 256,
        candidate_loss_weight: float = 0.0,
        candidate_sites_per_sample: int = 0,
        candidate_negatives_per_site: int = 0,
        candidate_max_prefix_tokens: int = 1536,
        candidate_keep_head_tokens: int = 256,
        **kwargs,
    ):
        self._group_by_length = group_by_length
        self.mem_bank = mem_bank or {}
        self.mem_dim = mem_dim
        self.max_slots = max_slots
        self.lr_lora = lr_lora
        self.lr_xattn = lr_xattn
        self.lr_gate = lr_gate
        self.lr_ff = lr_ff
        self.lr_gate_ff = lr_gate_ff
        self.lr_embed = lr_lora if lr_embed is None else lr_embed
        self.loss_chunk_t = loss_chunk_t
        self.candidate_loss_weight = float(candidate_loss_weight)
        self.candidate_sites_per_sample = int(candidate_sites_per_sample)
        self.candidate_negatives_per_site = int(candidate_negatives_per_site)
        self.candidate_max_prefix_tokens = int(candidate_max_prefix_tokens)
        self.candidate_keep_head_tokens = int(candidate_keep_head_tokens)
        self._last_debug_step = -1
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lora_params, embed_params = [], []
        attn_gate_params, ff_gate_params = [], []
        xattn_attn_params, xattn_ff_params = [], []
        other_trainables = []

        try:
            input_emb_param_ids = {id(p) for p in get_input_embeddings_module(self.model).parameters()}
        except Exception:
            input_emb_param_ids = set()

        try:
            output_emb = self.model.get_output_embeddings()
            output_emb_param_ids = {id(p) for p in output_emb.parameters()} if output_emb is not None else set()
        except Exception:
            output_emb_param_ids = set()

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            if id(p) in input_emb_param_ids or id(p) in output_emb_param_ids:
                embed_params.append(p)
            elif "lora_" in n:
                lora_params.append(p)
            elif n.endswith("attn_gate"):
                attn_gate_params.append(p)
            elif n.endswith("ff_gate"):
                ff_gate_params.append(p)
            elif "gated_cross_attn_layer.attn." in n:
                xattn_attn_params.append(p)
            elif "gated_cross_attn_layer.ff." in n:
                xattn_ff_params.append(p)
            else:
                other_trainables.append((n, p))

        if other_trainables:
            print("[WARN] Unexpected trainable params:")
            for n, _ in other_trainables[:20]:
                print("  -", n)

        opt_groups = []
        if lora_params:
            opt_groups.append({"params": lora_params, "lr": self.lr_lora})
        if embed_params:
            opt_groups.append({"params": embed_params, "lr": self.lr_embed})
        if attn_gate_params:
            opt_groups.append({"params": attn_gate_params, "lr": self.lr_gate})
        if ff_gate_params:
            opt_groups.append({"params": ff_gate_params, "lr": self.lr_gate_ff})
        if xattn_attn_params:
            opt_groups.append({"params": xattn_attn_params, "lr": self.lr_xattn})
        if xattn_ff_params:
            opt_groups.append({"params": xattn_ff_params, "lr": self.lr_ff})
        if other_trainables:
            opt_groups.append({"params": [p for _, p in other_trainables], "lr": self.lr_lora})

        try:
            from bitsandbytes.optim import PagedAdamW8bit
            self.optimizer = PagedAdamW8bit(opt_groups, weight_decay=0.0)
        except Exception:
            self.optimizer = torch.optim.AdamW(opt_groups, weight_decay=0.0)

        print(
            f"[OPT] param groups: "
            f"lora={sum(p.numel() for p in lora_params):,} "
            f"embed={sum(p.numel() for p in embed_params):,} "
            f"attn_gate={sum(p.numel() for p in attn_gate_params):,} "
            f"ff_gate={sum(p.numel() for p in ff_gate_params):,} "
            f"xattn_attn={sum(p.numel() for p in xattn_attn_params):,} "
            f"xattn_ff={sum(p.numel() for p in xattn_ff_params):,} "
            f"lr_lora={self.lr_lora:g} "
            f"lr_embed={self.lr_embed:g} "
            f"lr_gate={self.lr_gate:g} "
            f"lr_gate_ff={self.lr_gate_ff:g} "
            f"lr_xattn={self.lr_xattn:g} "
            f"lr_ff={self.lr_ff:g}"
        )
        return self.optimizer

    def get_train_dataloader(self):
        if not self._group_by_length:
            return super().get_train_dataloader()

        sampler = LengthGroupedSampler(
            self.args.train_batch_size,
            dataset=self.train_dataset,
            lengths=getattr(self.train_dataset, "lengths", None),
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=(self.args.dataloader_num_workers > 0),
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        # IMPORTANT: clear only after backward, so checkpoint recomputation
        # still sees the conditioned HARP state
        if hasattr(model, "clear_harp"):
            model.clear_harp()

        if self.accelerator.sync_gradients and self.state.global_step != self._last_debug_step:
            if self.state.global_step % 20 == 0:
                print_xattn_gate_stats(model, print_grads=True)
                if hasattr(model, "clear_harp") and not getattr(self.args, "disable_harp", False):
                    print_xattn_forward_stats(model)
                self._last_debug_step = self.state.global_step

        return loss.detach() / self.args.gradient_accumulation_steps
    

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        has_labels = "labels" in inputs and inputs["labels"] is not None

        if has_labels:
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            loss = loss.mean().detach()

            if prediction_loss_only:
                return (loss, None, None)

            logits = outputs.logits.detach() if hasattr(outputs, "logits") else outputs[0].detach()
            labels = inputs["labels"].detach()
            return (loss, logits, labels)

        model_inputs = dict(inputs)
        for k in (
            "sample_weight",
            "kernel_name",
            "routing_start_idx",
            "token_weights",
            "xattn_apply_mask",
            "contrastive_sites",
        ):
            model_inputs.pop(k, None)

        with torch.no_grad():
            outputs = model(**model_inputs)

        logits = outputs.logits.detach() if hasattr(outputs, "logits") else outputs[0].detach()
        return (None, logits, None)


    def _condition_harp_from_kernel_names(self, model, kernel_names: List[str]):
        kvs, ms = [], []
        for k in kernel_names:
            pack = self.mem_bank.get(k) or self.mem_bank.get(normalize_kname(k))
            if pack is None:
                kvs.append(torch.zeros((self.max_slots, self.mem_dim), dtype=torch.float32))
                ms.append(torch.zeros((self.max_slots,), dtype=torch.bool))
            else:
                kvs.append(pack["kv"])
                ms.append(pack["mask"])
        mem_kv = torch.stack(kvs, dim=0)  # [B, S, mem_dim]
        mem_m  = torch.stack(ms, dim=0)   # [B, S]
        device = next(model.parameters()).device
        model.condition_harp(mem_kv.to(device), mem_m.to(device))

    def truncate_scoring_prefix_preserve_target(
        self,
        prefix_ids: List[int],
        routing_start_idx: Optional[int],
        max_prefix_tokens: int,
        keep_head_tokens: int,
    ) -> Tuple[List[int], Optional[int]]:
        if (
            routing_start_idx is None
            or max_prefix_tokens <= 0
            or len(prefix_ids) <= max_prefix_tokens
        ):
            return prefix_ids, routing_start_idx

        R = int(routing_start_idx)
        prompt_ids = prefix_ids[:R]
        target_prefix_ids = prefix_ids[R:]

        # Always preserve the entire generated target prefix if possible,
        # because HARP routing depends on target anchors already emitted.
        if len(target_prefix_ids) >= max_prefix_tokens:
            kept_target = target_prefix_ids[-max_prefix_tokens:]
            return kept_target, 0

        prompt_budget = max_prefix_tokens - len(target_prefix_ids)
        if len(prompt_ids) <= prompt_budget:
            return prefix_ids, R

        keep_head = min(max(0, keep_head_tokens), max(0, prompt_budget - 1))
        keep_tail = prompt_budget - keep_head

        if keep_head <= 0:
            kept_prompt = prompt_ids[-prompt_budget:]
        else:
            kept_prompt = prompt_ids[:keep_head] + prompt_ids[-keep_tail:]

        new_prefix = kept_prompt + target_prefix_ids
        new_routing_start_idx = len(kept_prompt)
        return new_prefix, new_routing_start_idx


    def _score_candidate_sequence(
        self,
        model,
        prefix_ids: List[int],
        candidate_ids: List[int],
        routing_start_idx: Optional[int],
        use_harp: bool,
    ):
        device = next(model.parameters()).device

        effective_route_idx = int(routing_start_idx) if routing_start_idx is not None else None

        prefix_ids, effective_route_idx = self.truncate_scoring_prefix_preserve_target(
            prefix_ids=prefix_ids,
            routing_start_idx=effective_route_idx,
            max_prefix_tokens=self.candidate_max_prefix_tokens,
            keep_head_tokens=self.candidate_keep_head_tokens,
        )

        base_input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        base_attention_mask = torch.ones_like(base_input_ids)

        cand_tensor = torch.tensor([candidate_ids], dtype=torch.long, device=device)

        full_input_ids = torch.cat([base_input_ids, cand_tensor], dim=1)
        full_attention_mask = torch.ones_like(full_input_ids)

        model_inputs = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
        }

        if use_harp:
            if effective_route_idx is None:
                raise ValueError("effective_route_idx is required when use_harp=True")

            model_inputs["routing_start_idx"] = torch.tensor(
                [effective_route_idx],
                dtype=torch.long,
                device=device,
            )
            xmask = torch.zeros(
                (1, full_input_ids.shape[1]),
                dtype=torch.float32,
                device=device,
            )
            xmask[:, effective_route_idx:] = 1.0
            model_inputs["xattn_apply_mask"] = xmask

        outputs = model(**model_inputs)

        base_len = len(prefix_ids)
        cand_len = len(candidate_ids)

        cand_logits = outputs.logits[:, base_len - 1: base_len - 1 + cand_len, :].float()
        token_logprobs = F.log_softmax(cand_logits, dim=-1)
        token_logprobs = token_logprobs.gather(
            -1,
            cand_tensor.unsqueeze(-1)
        ).squeeze(-1).squeeze(0)

        return token_logprobs.mean()

    def _compute_candidate_loss(
        self,
        model,
        contrastive_sites,
        kernel_names,
        routing_start_idx,
        sample_weights=None,
    ):
        """
        The model should learn:
            score(gold RHS) > score(negative RHS alternatives)

        We compute:
        1) mean candidate loss per sample (over its available selected sites)
        2) weighted average across samples, matching the CE loss logic
        """
        device = next(model.parameters()).device

        per_sample_losses = []
        per_sample_weights = []

        harp_enabled = hasattr(model, "condition_harp") and getattr(model, "initialized_harp_flamingo", False)

        for b_idx, sites in enumerate(contrastive_sites):
            if not sites:
                continue

            route_idx = int(routing_start_idx[b_idx].item()) if routing_start_idx is not None else 0

            if harp_enabled:
                self._condition_harp_from_kernel_names(model, [kernel_names[b_idx]])

            site_losses = []

            for site in sites[:self.candidate_sites_per_sample]:
                neg_ids_list = site["negative_ids"][:self.candidate_negatives_per_site]
                if not neg_ids_list:
                    continue

                gold_score = self._score_candidate_sequence(
                    model=model,
                    prefix_ids=site["prefix_ids"],
                    candidate_ids=site["gold_ids"],
                    routing_start_idx=route_idx,
                    use_harp=harp_enabled,
                )

                neg_scores = [
                    self._score_candidate_sequence(
                        model=model,
                        prefix_ids=site["prefix_ids"],
                        candidate_ids=neg_ids,
                        routing_start_idx=route_idx,
                        use_harp=harp_enabled,
                    )
                    for neg_ids in neg_ids_list
                ]

                scores = torch.stack([gold_score] + neg_scores, dim=0)
                site_loss = -F.log_softmax(scores, dim=0)[0]
                site_losses.append(site_loss)

            if not site_losses:
                continue

            sample_loss = torch.stack(site_losses).mean()
            per_sample_losses.append(sample_loss)

            if sample_weights is not None:
                per_sample_weights.append(sample_weights[b_idx].to(device=device, dtype=torch.float32))
            else:
                per_sample_weights.append(torch.tensor(1.0, device=device, dtype=torch.float32))

        if not per_sample_losses:
            return torch.zeros((), device=device, dtype=torch.float32)

        per_sample_losses = torch.stack(per_sample_losses)
        per_sample_weights = torch.stack(per_sample_weights)

        return (per_sample_losses * per_sample_weights).sum() / per_sample_weights.sum().clamp(min=1e-8)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        weights = inputs.pop("sample_weight", None)
        kernel_names = inputs.pop("kernel_name", None)
        labels = inputs["labels"]
        token_weights = inputs.pop("token_weights", None)
        xattn_apply_mask = inputs.pop("xattn_apply_mask", None)
        contrastive_sites = inputs.pop("contrastive_sites", None)

        if kernel_names is not None and hasattr(model, "condition_harp"):
            self._condition_harp_from_kernel_names(model, kernel_names)

        try:
            routing_start_idx = inputs.pop("routing_start_idx", None)

            model_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

            harp_enabled = hasattr(model, "condition_harp") and getattr(model, "initialized_harp_flamingo", False)

            if harp_enabled:
                if routing_start_idx is not None:
                    model_inputs["routing_start_idx"] = routing_start_idx
                if xattn_apply_mask is not None:
                    model_inputs["xattn_apply_mask"] = xattn_apply_mask

            outputs = model(**model_inputs)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_token_weights = token_weights[:, 1:].contiguous() if token_weights is not None else None

            B, Tm1 = shift_labels.shape
            device = shift_labels.device

            chunk_t = int(self.loss_chunk_t)
            loss_sum = torch.zeros(B, device=device, dtype=torch.float32)
            tok_cnt = torch.zeros(B, device=device, dtype=torch.float32)

            for s in range(0, Tm1, chunk_t):
                e = min(Tm1, s + chunk_t)
                logits_chunk = shift_logits[:, s:e, :]
                labels_chunk = shift_labels[:, s:e]

                flat_logits = logits_chunk.reshape(-1, logits_chunk.size(-1))
                flat_labels = labels_chunk.reshape(-1)

                per_tok = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction="none",
                ).view(B, -1)

                mask = labels_chunk.ne(-100)
                tok_weight = torch.ones_like(per_tok)

                # Each supervised token is weighted by : 
                # 0 for prompt / fixed schema , directive-kind-specific weight for RHS tokens , value_w for EOS
                if shift_token_weights is not None:
                    tok_weight = tok_weight * shift_token_weights[:, s:e].to(per_tok.dtype)

                weighted_mask = tok_weight * mask.to(tok_weight.dtype)
                loss_sum += (per_tok * weighted_mask).sum(dim=1)
                tok_cnt += weighted_mask.sum(dim=1)

            # Each example gets normalized by its weighted token count
            # Long targets do not automatically dominate short ones
            per_ex = loss_sum / tok_cnt.clamp(min=1.0)
 
            # Change the relative contribution to loss of each example
            if weights is not None:
                w = weights.to(device=device, dtype=per_ex.dtype)
                ce_loss = (per_ex * w).sum() / w.sum().clamp(min=1e-8)
            else:
                ce_loss = per_ex.mean()

            # contrastive loss : active only during training to make eval lighter
            cand_loss = torch.zeros((), device=device, dtype=torch.float32)
            if (
                model.training
                and self.candidate_loss_weight > 0.0
                and contrastive_sites is not None
                and self.candidate_sites_per_sample > 0
                and self.candidate_negatives_per_site > 0
            ):
                cand_loss = self._compute_candidate_loss(
                    model=model,
                    contrastive_sites=contrastive_sites,
                    kernel_names=kernel_names,
                    routing_start_idx=routing_start_idx,
                    sample_weights=weights,
                )

            # CE loss : predict the exact gold RHS tokens
            # Contrastive loss :  among plausible alternatives for the same site, prefer the gold RHS over nearby hard negatives
            loss = ce_loss + self.candidate_loss_weight * cand_loss
            return (loss, outputs) if return_outputs else loss
    
        finally:
            if hasattr(model, "clear_harp") and not model.training:
                model.clear_harp()



# ===============================================
# Stage Configs (Stage_1 / Stage_2)
# ===============================================
@dataclass
class StageRunConfig:
    name: str
    output_dir: str
    disable_harp: bool

    init_adapter_dir: str = ""
    init_harp_xattn_from: str = ""
    best_dir_name: str = "best_custom_stage1"

    value_loss_weight: float = 1.0

    lr_lora: float = 5e-5
    lr_embed: float = 5e-5
    lr_xattn: float = 0.0
    lr_gate: float = 0.0
    lr_ff: float = 0.0
    lr_gate_ff: float = 0.0

    epochs: int = 2
    max_steps: int = -1
    eval_steps: int = 100
    save_steps: int = 100


def clone_args(args):
    return argparse.Namespace(**vars(args))


def make_stage_args(base_args, cfg: StageRunConfig):
    a = clone_args(base_args)

    a.output_dir = cfg.output_dir
    a.best_dir_name = cfg.best_dir_name
    a.disable_harp = cfg.disable_harp

    a.init_adapter_dir = cfg.init_adapter_dir
    a.init_harp_xattn_from = cfg.init_harp_xattn_from
    a.resume_from_checkpoint = ""

    a.value_loss_weight = cfg.value_loss_weight

    a.lr_lora = cfg.lr_lora
    a.lr_embed = cfg.lr_embed
    a.lr_xattn = cfg.lr_xattn
    a.lr_gate = cfg.lr_gate
    a.lr_ff = cfg.lr_ff
    a.lr_gate_ff = cfg.lr_gate_ff

    a.epochs = cfg.epochs
    a.max_steps = cfg.max_steps
    a.eval_steps = cfg.eval_steps
    a.save_steps = cfg.save_steps

    return a


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_default_stage_configs(args):
    # Stage 1: learn objective-conditioned RHS values only
    stage1 = StageRunConfig(
        name="stage1_goal_rhs_only_sft",
        output_dir=args.stage1_output_dir,
        best_dir_name="best_custom_stage1",
        disable_harp=True,

        value_loss_weight=1.0,

        lr_lora=args.lr_lora,
        lr_embed=args.lr_embed,
        lr_xattn=0.0,
        lr_gate=0.0,
        lr_ff=0.0,
        lr_gate_ff=0.0,

        epochs=args.epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )

    # Stage 2: keep same deterministic target-side format, enable HARP for RHS refinement
    stage2 = StageRunConfig(
        name="stage2_goal_harp_rhs_only",
        output_dir=args.stage2_output_dir,
        disable_harp=False,

        init_harp_xattn_from="",
        init_adapter_dir=os.path.join(args.stage1_output_dir, "best_custom_stage1"),
        best_dir_name="best_custom_stage2",

        value_loss_weight=1.0,

        lr_lora=0.0,
        lr_embed=0.0,
        lr_xattn=args.stage2_lr_xattn,
        lr_gate=args.stage2_lr_gate,
        lr_ff=args.stage2_lr_ff,
        lr_gate_ff=args.stage2_lr_gate_ff,

        epochs=args.stage2_epochs,
        max_steps=args.stage2_max_steps,
        eval_steps=args.stage2_eval_steps,
        save_steps=args.stage2_save_steps,
    )

    return stage1, stage2



# =================================
# Main Training
# =================================
def run_single_training(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    dump_root = os.path.join(args.output_dir, "selected_debug")
    os.makedirs(dump_root, exist_ok=True)

    rows = load_rows(args.dataset)
    print(f"[INFO] Loaded {len(rows)} raw rows from {args.dataset}")
    fam_counts = Counter(r["_family"] for r in rows)
    print("[INFO] Raw rows per family (top 15):", fam_counts.most_common(15))

    if args.split_json:
        split_spec = load_split_spec(args.split_json)
        raw_train_rows, raw_val_rows, raw_test_rows = apply_split_spec(rows, split_spec)
        print(f"[INFO] Loaded split from {args.split_json}")
    elif args.split_mode == "family":
        val_fams = {normalize_name(x) for x in args.val_families.split(";") if x.strip()}
        test_fams = {normalize_name(x) for x in args.test_families.split(";") if x.strip()}
        print("[INFO] val_families:", sorted(val_fams))
        print("[INFO] test_families:", sorted(test_fams))
        raw_train_rows, raw_val_rows, raw_test_rows = split_by_family(rows, val_fams, test_fams)
    else:
        raw_train_rows, raw_val_rows, raw_test_rows = split_rows_random_design(
            rows,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
            stratify_by_kernel=args.stratify_by_kernel,
        )
        print(f"[INFO] random design-point split with val_ratio={args.val_ratio}, test_ratio={args.test_ratio}, split_seed={args.split_seed}, stratify_by_kernel={args.stratify_by_kernel}")

    print(f"[INFO] Raw split sizes: train={len(raw_train_rows)} val={len(raw_val_rows)} test={len(raw_test_rows)}")

    if args.save_split_json:
        save_split_spec(args.save_split_json, raw_train_rows, raw_val_rows, raw_test_rows)
        print(f"[INFO] Saved split spec -> {args.save_split_json}")

    goal_key = GOALS[args.objective]["tag"]

    train_rows, train_goal_info = select_goal_rows(
        raw_train_rows,
        goal_mode=args.objective,
        top_k=args.top_k,
        domination_penalty=args.goal_domination_penalty,
        max_dominated_gap=args.goal_max_dominated_gap,
        score_weight_min=args.score_weight_min,
        score_weight_power=args.score_weight_power,
    )

    val_rows, val_goal_info = select_goal_rows(
        raw_val_rows,
        goal_mode=args.objective,
        top_k=args.top_k,
        domination_penalty=args.goal_domination_penalty,
        max_dominated_gap=args.goal_max_dominated_gap,
        score_weight_min=args.score_weight_min,
        score_weight_power=args.score_weight_power,
    )

    test_rows, test_goal_info = select_goal_rows(
        raw_test_rows,
        goal_mode=args.objective,
        top_k=args.top_k,
        domination_penalty=args.goal_domination_penalty,
        max_dominated_gap=args.goal_max_dominated_gap,
        score_weight_min=args.score_weight_min,
        score_weight_power=args.score_weight_power,
    )

    print(f"[INFO] Selected split sizes: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    dump_jsonl(os.path.join(dump_root, f"train_selected_{goal_key}.jsonl"), train_rows)
    dump_json(os.path.join(dump_root, f"train_selected_{goal_key}.indices.json"), train_goal_info)
    if val_rows:
        dump_jsonl(os.path.join(dump_root, f"val_selected_{goal_key}.jsonl"), val_rows)
        dump_json(os.path.join(dump_root, f"val_selected_{goal_key}.indices.json"), val_goal_info)
    if test_rows:
        dump_jsonl(os.path.join(dump_root, f"test_selected_{goal_key}.jsonl"), test_rows)
        dump_json(os.path.join(dump_root, f"test_selected_{goal_key}.indices.json"), test_goal_info)

    rhs_candidate_bank = build_rhs_candidate_bank(train_rows)
    print("[INFO] RHS candidate bank sizes:", {k: len(v) for k, v in sorted(rhs_candidate_bank.items())})

    selection_cases = build_selection_cases(
        val_rows,
        goal_mode=args.objective,
        max_kernels=args.selection_num_val_kernels,
        min_coverage=args.min_site_coverage,
        min_supervised_sites=args.min_supervised_sites,
    )
    print(f"[INFO] Built {len(selection_cases)} validation selection cases from {args.selection_num_val_kernels} kernels")

    if args.disable_harp:
        mem_bank = {}
        print("[INFO] HARP disabled -> skipping memory bank loading")
    else:
        mem_bank = load_memory_bank(args.memory_dir)
        print(f"[INFO] Memory bank keys: {len(mem_bank)}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    special_tokens = [g["token"] for g in GOALS.values()] + SOURCE_PLACEHOLDER_TOKENS + TARGET_PLACEHOLDER_TOKENS
    tok.add_special_tokens({"additional_special_tokens": special_tokens})

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    prompt_template = PROMPT_TEMPLATE

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
    )

    base.resize_token_embeddings(len(tok))
    if hasattr(base.config, "tie_word_embeddings"):
        base.config.tie_word_embeddings = True

    try:
        base.tie_weights()
    except Exception as e:
        print(f"[WARN] tie_weights() failed: {e}")

    base.config.use_cache = False

    gc_kwargs = {"use_reentrant": False} if args.gradient_checkpointing else None
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    resume_ckpt = os.path.abspath(args.resume_from_checkpoint) if args.resume_from_checkpoint else ""
    init_adapter_dir = os.path.abspath(args.init_adapter_dir) if args.init_adapter_dir else ""

    if resume_ckpt and os.path.isdir(resume_ckpt):
        model = PeftModel.from_pretrained(
            base,
            resume_ckpt,
            is_trainable=True,
        )
        print(f"[INIT] Loaded PEFT adapter from resume checkpoint: {resume_ckpt}")

    elif init_adapter_dir and os.path.isdir(init_adapter_dir):
        model = PeftModel.from_pretrained(
            base,
            init_adapter_dir,
            is_trainable=True,
        )
        print(f"[INIT] Loaded adapter from: {init_adapter_dir}")

    else:
        model = get_peft_model(base, lora_cfg)
        print("[INIT] Created fresh LoRA adapter")

    if not args.disable_harp:
        extend_instance(model, HARPLMMixin)
        decoder_layers_attr_name = infer_decoder_layers_attr_name(model)
        model.set_decoder_layers_attr_name(decoder_layers_attr_name)

        placeholder_token_ids = tok.convert_tokens_to_ids(TARGET_PLACEHOLDER_TOKENS)
        hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Could not infer LM hidden size from model.config")

        model.init_harp_flamingo(
            placeholder_token_ids=placeholder_token_ids,
            lang_hidden_size=hidden_size,
            mem_hidden_size=args.mem_dim,
            cross_attn_every_n_layers=args.every_n_layers,
            gradient_checkpointing=args.gradient_checkpointing,
            xattn_heads=args.xattn_heads,
            xattn_dim_head=args.xattn_dim_head,
            xattn_ff_mult=args.xattn_ff_mult,
            only_attend_immediate_memory=True,
            mask_mode="segment",
        )

        print(f"[HARP-XATTN] decoder_layers_attr_name={decoder_layers_attr_name}")
        print(f"[HARP-XATTN] inserted gated xattn every {args.every_n_layers} decoder layers")
        move_harp_modules_to_model_device(model)
    else:
        print("[HARP-XATTN] disabled for Stage 1 format-only training")

    if resume_ckpt and os.path.isdir(resume_ckpt):
        resume_harp_xattn = os.path.join(resume_ckpt, "harp_xattn.pt")
        load_partial_harp_xattn(model, resume_harp_xattn, tag="HARP-RESUME")
    elif args.init_harp_xattn_from:
        load_partial_harp_xattn(model, args.init_harp_xattn_from, tag="HARP-INIT")

    input_emb = unfreeze_input_embeddings(model)
    print(f"[TOKENS] input embeddings unfrozen: {input_emb.weight.requires_grad}")

    model.print_trainable_parameters()

    try:
        model.tie_weights()
    except Exception:
        pass

    inp = model.get_input_embeddings().weight
    out = model.get_output_embeddings().weight
    print("tied =", inp.data_ptr() == out.data_ptr())

    def enable_only_selected_rows(weight: torch.nn.Parameter, token_ids):
        weight.requires_grad_(True)
        token_ids = torch.tensor(sorted(set(token_ids)), dtype=torch.long)

        def grad_mask_hook(grad):
            mask = torch.zeros(grad.size(0), device=grad.device, dtype=grad.dtype)
            mask[token_ids.to(grad.device)] = 1.0
            return grad * mask.unsqueeze(1)

        weight.register_hook(grad_mask_hook)

    special_ids = tok.convert_tokens_to_ids(
        [g["token"] for g in GOALS.values()] + SOURCE_PLACEHOLDER_TOKENS + TARGET_PLACEHOLDER_TOKENS
    )

    inp_emb = model.get_input_embeddings()
    enable_only_selected_rows(inp_emb.weight, special_ids)

    out_emb = model.get_output_embeddings()
    if out_emb is not None and out_emb.weight is not inp_emb.weight:
        enable_only_selected_rows(out_emb.weight, special_ids)

    train_ds = SFTDataset(
        train_rows,
        tok,
        args.max_length,
        value_loss_weight=args.value_loss_weight,
        candidate_sites_per_sample=args.candidate_sites_per_sample,
        candidate_negatives_per_site=args.candidate_negatives_per_site,
    )

    val_ds = SFTDataset(
        val_rows,
        tok,
        args.max_length,
        value_loss_weight=args.value_loss_weight,
        candidate_sites_per_sample=args.candidate_sites_per_sample,
        candidate_negatives_per_site=args.candidate_negatives_per_site,
    ) if val_rows else None

    collator = PadCollator(tok)

    steps_per_epoch = math.ceil(len(train_ds) / max(1, args.batch_size))
    total_steps = int(steps_per_epoch * args.epochs / max(1, args.grad_accum))
    effective_total_steps = args.max_steps if args.max_steps > 0 else total_steps
    warmup_steps = int(0.03 * effective_total_steps)


    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=max(args.lr_lora, args.lr_xattn, args.lr_gate, args.lr_embed),
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=False,
        save_total_limit=6,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],   # <- add this
    )


    trainer = LengthGroupedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        group_by_length=args.group_by_length,
        mem_bank=mem_bank,
        mem_dim=args.mem_dim,
        max_slots=args.max_slots,
        lr_lora=args.lr_lora,
        lr_xattn=args.lr_xattn,
        lr_embed=args.lr_embed,
        lr_gate=args.lr_gate,
        lr_ff=args.lr_ff,
        lr_gate_ff=args.lr_gate_ff,
        loss_chunk_t=args.loss_chunk_t,
        candidate_loss_weight=args.candidate_loss_weight,
        candidate_sites_per_sample=args.candidate_sites_per_sample,
        candidate_negatives_per_site=args.candidate_negatives_per_site,
        candidate_max_prefix_tokens=args.candidate_max_prefix_tokens,
        candidate_keep_head_tokens=args.candidate_keep_head_tokens,
    )

    if not args.disable_harp:
        trainer.add_callback(SaveHarpXattnCallback())

    if selection_cases:
        trainer.add_callback(
            StageValSelectionCallback(
                tokenizer=tok,
                selection_cases=selection_cases,
                rhs_candidate_bank=rhs_candidate_bank,
                output_dir=args.output_dir,
                max_prompt_tokens=args.max_length,
                candidate_score_reduction="mean",
                best_dir_name=args.best_dir_name,
                mem_bank=mem_bank,
                mem_dim=args.mem_dim,
                max_slots=args.max_slots,
            )
        )

    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        print(f"[INFO] Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        print(f"[INFO] No checkpoint found. Starting from scratch.")
        trainer.train()

    best_dir = os.path.join(args.output_dir, args.best_dir_name)
    if not os.path.isdir(best_dir):
        os.makedirs(best_dir, exist_ok=True)
        trainer.save_model(best_dir)
        tok.save_pretrained(best_dir)
        if not args.disable_harp:
            torch.save(get_harp_xattn_state_dict(model), os.path.join(best_dir, "harp_xattn.pt"))

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if not args.disable_harp:
        torch.save(
            get_harp_xattn_state_dict(model),
            os.path.join(args.output_dir, "harp_xattn.pt")
        )
        print(f"[DONE] Saved LoRA + HARP xattn adapters to: {args.output_dir}")
    else:
        print(f"[DONE] Saved LoRA adapter to: {args.output_dir}")



def main():
    ap = argparse.ArgumentParser()

    # Data / Memory / Model
    ap.add_argument("--dataset", type=str, default="/home/elvouvali/LLM_data/all_kernels_llm_data_multi_target.jsonl")
    ap.add_argument("--memory_dir", type=str, default="/home/elvouvali/save/harp/memory_tokens/")
    ap.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-7b-base")
    ap.add_argument("--objective", type=str, required=True, choices=GOAL_ORDER)

    # Split Mode
    ap.add_argument("--split_mode", type=str, default="family", choices=["family", "random_design"])
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--stratify_by_kernel", action="store_true")
    ap.add_argument("--split_json", type=str, default="")
    ap.add_argument("--save_split_json", type=str, default="")

    # Goal-specific point selection
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--goal_domination_penalty", type=float, default=0.25)
    ap.add_argument("--goal_max_dominated_gap", type=float, default=0.12)
    ap.add_argument("--candidate_loss_weight", type=float, default=0.0) # controls the influence of the contrastive loss in the final loss
    ap.add_argument("--candidate_sites_per_sample", type=int, default=2)
    ap.add_argument("--candidate_negatives_per_site", type=int, default=2)
    ap.add_argument("--candidate_max_prefix_tokens", type=int, default=1536)
    ap.add_argument("--candidate_keep_head_tokens", type=int, default=256)
    ap.add_argument("--val_families", type=str, default="rodinia_pathfinder;machsuite_sort_radix")
    ap.add_argument("--test_families", type=str, default="serrano-kalman-filter")
    ap.add_argument("--min_supervised_sites", type=int, default=2)
    ap.add_argument("--min_site_coverage", type=float, default=0.85)
    ap.add_argument("--score_weight_min", type=float, default=0.6)
    ap.add_argument("--score_weight_power", type=float, default=1.0)

    # Training Params
    ap.add_argument("--max_length", type=int, default=7168)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--group_by_length", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default="")
    ap.add_argument("--init_adapter_dir", type=str, default="")
    ap.add_argument("--init_harp_xattn_from", type=str, default="")
    ap.add_argument("--value_loss_weight", type=float, default=1.0)

    # LoRA
    ap.add_argument("--lr_lora", type=float, default=5e-5)
    ap.add_argument("--lr_embed", type=float, default=5e-5)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # HARP Memory
    ap.add_argument("--mem_dim", type=int, default=32)
    ap.add_argument("--max_slots", type=int, default=64)
    ap.add_argument("--every_n_layers", type=int, default=8)
    ap.add_argument("--xattn_heads", type=int, default=4)
    ap.add_argument("--xattn_dim_head", type=int, default=64)
    ap.add_argument("--xattn_ff_mult", type=int, default=1)
    ap.add_argument("--lr_xattn", type=float, default=0.0)
    ap.add_argument("--lr_gate", type=float, default=0.0)
    ap.add_argument("--lr_ff", type=float, default=0.0)
    ap.add_argument("--lr_gate_ff", type=float, default=0.0)

    # Best Checkpoint Selection
    ap.add_argument("--selection_num_val_kernels", type=int, default=4)
    ap.add_argument("--best_dir_name", type=str, default="best_custom_stage1")

    # Trainer / pipeline
    ap.add_argument("--disable_harp", action="store_true")
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--loss_chunk_t", type=int, default=256)
    ap.add_argument("--run_mode", type=str, default="two_stage", choices=["single", "two_stage"])
    ap.add_argument("--output_dir", type=str, default="")
    ap.add_argument("--stage1_output_dir", type=str, default="")
    ap.add_argument("--stage2_output_dir", type=str, default="")
    ap.add_argument("--stage2_epochs", type=int, default=4)
    ap.add_argument("--stage2_max_steps", type=int, default=-1)
    ap.add_argument("--stage2_eval_steps", type=int, default=50)
    ap.add_argument("--stage2_save_steps", type=int, default=50)
    ap.add_argument("--stage2_lr_xattn", type=float, default=1e-4)
    ap.add_argument("--stage2_lr_gate", type=float, default=2e-4)
    ap.add_argument("--stage2_lr_ff", type=float, default=0.0)
    ap.add_argument("--stage2_lr_gate_ff", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    goal_tag = GOALS[args.objective]["tag"]
    if not args.stage1_output_dir:
        args.stage1_output_dir = f"./sft_harp_xattn_{goal_tag}_stage1"
    if not args.stage2_output_dir:
        args.stage2_output_dir = f"./sft_harp_xattn_{goal_tag}_stage2"

    if args.run_mode == "single":
        if not args.output_dir:
            args.output_dir = args.stage1_output_dir
        run_single_training(args)
        return

    stage1_cfg, stage2_cfg = build_default_stage_configs(args)

    print("\n" + "=" * 120)
    print(f"[PIPELINE] Running {stage1_cfg.name} for {args.objective}")
    print(f"[PIPELINE] output_dir={stage1_cfg.output_dir}")
    print("=" * 120)
    stage1_args = make_stage_args(args, stage1_cfg)
    run_single_training(stage1_args)

    stage1_best_dir = os.path.join(stage1_cfg.output_dir, stage1_cfg.best_dir_name)
    if not os.path.isdir(stage1_best_dir):
        raise FileNotFoundError(
            f"Stage 1 best adapter was not created: {stage1_best_dir}"
        )

    cleanup_cuda()

    print("\n" + "=" * 120)
    print(f"[PIPELINE] Running {stage2_cfg.name} for {args.objective}")
    print(f"[PIPELINE] output_dir={stage2_cfg.output_dir}")
    print(f"[PIPELINE] init_adapter_dir={stage2_cfg.init_adapter_dir}")
    print("=" * 120)

    stage2_args = make_stage_args(args, stage2_cfg)

    if args.save_split_json:
        stage2_args.split_json = args.save_split_json
        stage2_args.save_split_json = ""

    run_single_training(stage2_args)
    


if __name__ == "__main__":
    main()
