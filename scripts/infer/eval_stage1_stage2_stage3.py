import argparse
import json
import os
import re
import warnings
import math
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops_exts import rearrange_many
from peft import PeftModel
from torch import einsum
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================
# Prompt / objective tokens
# ============================================================
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

OBJ_TOKENS = {
    "PARETO_LATENCY_EXTREME": "<OBJ=PARETO_LATENCY_EXTREME>",
    "PARETO_KNEE": "<OBJ=PARETO_KNEE>",
    "PARETO_AREA_EXTREME": "<OBJ=PARETO_AREA_EXTREME>",
}

TARGET_PLACEHOLDER_TOKENS = [f"<L{i}>" for i in range(1, 65)]
SOURCE_PLACEHOLDER_TOKENS = [f"<SRC_L{i}>" for i in range(1, 65)]


# ============================================================
# Regexes
# ============================================================
SOURCE_LABEL_RE = re.compile(
    r"^\s*(?:/\*\s*(L\d+)\s*:\s*\*/|(L\d+)\s*:)",
    re.IGNORECASE,
)

TARGET_LINE_LABEL_RE = re.compile(
    r"auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}\s*=",
    re.IGNORECASE,
)

ANCHOR_OR_ASSIGN_RE = re.compile(
    r"^\s*(<L\d+>|auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\}\s*=\s*.+)$",
    re.IGNORECASE | re.MULTILINE,
)

ASSIGN_RE = re.compile(
    r"^(auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\})\s*=\s*(.+)$",
    re.IGNORECASE,
)

SOURCE_PLACEHOLDER_IN_CODE_RE = re.compile(
    r"auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_(L\d+)\}",
    re.IGNORECASE,
)

LHS_KIND_RE = re.compile(
    r"^auto\{_([A-Z0-9]+(?:_[A-Z0-9]+)*)_L\d+\}$",
    re.IGNORECASE,
)


# ============================================================
# Small helpers
# ============================================================
def normalize_name(s: str) -> str:
    return re.sub(r"[-\s]+", "_", s.strip().lower())


def normalize_kname(s: str) -> str:
    return normalize_name(s).replace("-", "_")


def mode_from_weights(w_lat: float, w_area: float) -> str:
    eps = 1e-9
    if abs(w_lat - 1.0) < eps and abs(w_area - 0.0) < eps:
        return "PARETO_LATENCY_EXTREME"
    if abs(w_lat - 0.0) < eps and abs(w_area - 1.0) < eps:
        return "PARETO_AREA_EXTREME"
    return "PARETO_KNEE"


def normalize_weight_pair(w_lat: float, w_area: float) -> Tuple[float, float]:
    s = w_lat + w_area
    if s <= 0:
        raise ValueError("w_lat + w_area must be > 0")
    return w_lat / s, w_area / s


def source_placeholder_token(label: str) -> str:
    return f"<SRC_{label.upper()}>"


def target_placeholder_token(label: str) -> str:
    return f"<{label.upper()}>"


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


def dump_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def dump_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# Source / target formatting helpers
# ============================================================
def replace_source_labels_with_tokens(text: str) -> str:
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
        grouped[m.group(1).upper()].append(line)

    out = []
    emitted = set()

    for label in label_order:
        if label in grouped:
            out.extend(grouped[label])
            emitted.add(label)

    for label in sorted(grouped.keys()):
        if label not in emitted:
            out.extend(grouped[label])

    out.extend(extras)
    return "\n".join(out)


def extract_ordered_lhs_plan(source_text: str) -> List[Tuple[str, str]]:
    by_label = defaultdict(list)

    for line in source_text.splitlines():
        for m in SOURCE_PLACEHOLDER_IN_CODE_RE.finditer(line):
            lhs = m.group(0)
            label = m.group(1).upper()
            if lhs not in by_label[label]:
                by_label[label].append(lhs)

    plan = []
    for label in extract_source_label_order(source_text):
        for lhs in by_label.get(label, []):
            plan.append((label, lhs))
    return plan


def build_rhs_map_from_target(target_text: str) -> Dict[str, str]:
    rhs_map = {}
    for raw_line in target_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = ASSIGN_RE.match(line)
        if m is None:
            continue
        rhs_map[m.group(1).strip()] = m.group(2).strip()
    return rhs_map


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
        m.group(0).strip() for m in ANCHOR_OR_ASSIGN_RE.finditer(text)
    ).strip()


def build_partial_deterministic_target_text(
    source_text: str,
    raw_target: str,
    min_supervised_sites: int = 1,
):
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


def build_prompt_for_case(code: str, obj_mode: str, w_lat: float, w_area: float) -> str:
    return PROMPT_TEMPLATE.format(
        obj_token=OBJ_TOKENS[obj_mode],
        code=replace_source_labels_with_tokens(code),
    )


# ============================================================
# Dataset / candidate bank
# ============================================================
@dataclass
class InferenceCase:
    kernel_name: str
    source_text: str
    obj_mode: str
    w_lat: float
    w_area: float
    reference_target: Optional[str] = None


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


def build_rhs_candidate_bank(rows: List[dict]) -> Dict[str, List[str]]:
    by_kind = defaultdict(set)

    for r in rows:
        target_core = reorder_target_by_source_order(r["input"], r["target"].strip())
        rhs_map = build_rhs_map_from_target(target_core)
        for lhs, rhs in rhs_map.items():
            rhs = rhs.strip()
            if rhs and rhs != "?":
                by_kind[lhs_kind(lhs)].add(rhs)

    fallbacks = {
        "PIPE": {"0", "1"},
        "UNROLL": {"0", "1", "2", "4", "8", "16", "32", "64"},
        "ARRAY_T": {"block", "cyclic", "complete"},
        "ARRAY_F": {"0", "1", "2", "4", "8", "16", "32", "64"},
        "ARRAY_D": {"1"},
    }

    out = {}
    all_kinds = set(by_kind.keys()) | set(fallbacks.keys())
    for kind in sorted(all_kinds):
        vals = set(by_kind.get(kind, set()))
        vals.update(fallbacks.get(kind, set()))
        out[kind] = sorted(vals, key=_rhs_sort_key)

    return out


def save_rhs_candidate_bank(path: str, bank: Dict[str, List[str]]):
    dump_json(path, bank)


def load_rhs_candidate_bank(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        bank = json.load(f)
    return {str(k): list(v) for k, v in bank.items()}


def get_rhs_candidates_for_lhs(lhs: str, rhs_candidate_bank: Dict[str, List[str]]) -> List[str]:
    kind = lhs_kind(lhs)
    cands = rhs_candidate_bank.get(kind, [])
    if not cands:
        raise KeyError(f"No RHS candidates found for lhs={lhs} kind={kind}")
    return cands


def maybe_filter_rows_for_candidate_bank(rows: List[dict], exclude_families: str) -> List[dict]:
    fams = {normalize_name(x) for x in exclude_families.split(";") if x.strip()}
    if not fams:
        return rows
    return [r for r in rows if normalize_name(r.get("_family", family_id_from_kernel_name(r["kernel_name"]))) not in fams]


def objective_from_case_dict(ex: dict) -> Tuple[str, float, float]:
    if "w_lat" in ex and "w_area" in ex:
        w_lat, w_area = normalize_weight_pair(float(ex["w_lat"]), float(ex["w_area"]))
        return mode_from_weights(w_lat, w_area), w_lat, w_area

    if "obj_mode" in ex:
        obj_mode = str(ex["obj_mode"]).strip().upper()

        if obj_mode == "PARETO_LATENCY_EXTREME":
            return obj_mode, 1.0, 0.0
        if obj_mode == "PARETO_AREA_EXTREME":
            return obj_mode, 0.0, 1.0
        if obj_mode == "PARETO_KNEE":
            return obj_mode, 0.5, 0.5

    if "objective" in ex:
        obj = str(ex["objective"]).strip().lower()

        if obj in {"pareto_latency_extreme", "latency_extreme", "latency", "min_lat", "min_latency"}:
            return "PARETO_LATENCY_EXTREME", 1.0, 0.0

        if obj in {"pareto_area_extreme", "area_extreme", "area", "min_area"}:
            return "PARETO_AREA_EXTREME", 0.0, 1.0

        if obj in {"pareto_knee", "knee", "balanced", "balance"}:
            return "PARETO_KNEE", 0.5, 0.5

        raise ValueError(f"Unknown objective: {obj}")

    raise ValueError("Case must provide either (w_lat, w_area), obj_mode, or objective")


def load_inference_cases_jsonl(path: str) -> List[InferenceCase]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            obj_mode, w_lat, w_area = objective_from_case_dict(ex)
            source_text = ex.get("input") or ex.get("code") or ex.get("source_text")
            if not source_text:
                raise ValueError("Each case must contain one of: input, code, source_text")

            ref = ex.get("reference_target")
            if ref is None and ex.get("target"):
                ref, _ = build_partial_deterministic_target_text(
                    source_text,
                    ex["target"],
                    min_supervised_sites=1,
                )

            rows.append(
                InferenceCase(
                    kernel_name=ex["kernel_name"],
                    source_text=source_text,
                    obj_mode=obj_mode,
                    w_lat=w_lat,
                    w_area=w_area,
                    reference_target=ref,
                )
            )
    return rows


# ============================================================
# HARP memory bank
# ============================================================
def load_memory_bank(memory_dir: str) -> Dict[str, dict]:
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
                assert lbl == j + 1, f"{fn}: non-contiguous labels {[x[0] for x in active]}"
                dense_kv[j] = vec
                dense_mask[j] = True

            kv = dense_kv.contiguous()
            mask = dense_mask.contiguous()

        bank[k] = {"kv": kv.contiguous(), "mask": mask.contiguous()}
        bank[normalize_kname(k)] = bank[k]

    return bank


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


# ============================================================
# HARP / Flamingo utilities
# ============================================================
def extend_instance(obj, mixin):
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (mixin, base_cls), {})


def getattr_recursive(obj, att):
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def get_first_real_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0")


def align_module_to_hidden_dtype(module: nn.Module, hidden_states: torch.Tensor):
    ref_device = hidden_states.device
    ref_dtype = hidden_states.dtype

    p = next((p for p in module.parameters() if p.is_floating_point()), None)
    if p is None:
        return

    if p.device != ref_device or p.dtype != ref_dtype:
        module.to(device=ref_device, dtype=ref_dtype)

def move_harp_modules_to_model_device(model):
    device = get_first_real_device(model)
    moved = 0
    for module in model.modules():
        if isinstance(module, GatedCrossAttentionBlock):
            module.to(device=device)
            moved += 1
    print(f"[HARP-DEVICE] moved {moved} HARP blocks to {device}")


def print_xattn_forward_stats(model):
    found = False
    for name, module in model.named_modules():
        if isinstance(module, MaskedCrossAttention) and getattr(module, "last_debug", None):
            found = True
            print(f"[XATTN-DBG] {name}: {module.last_debug}")
    if not found:
        print("[XATTN-DBG] no cross-attn forward stats collected yet")


def get_harp_xattn_state_dict(model):
    sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in sd.items() if "gated_cross_attn_layer" in k}


def load_partial_harp_xattn(model, harp_xattn_path: str, tag: str):
    if not harp_xattn_path or not os.path.isfile(harp_xattn_path):
        print(f"[{tag}] no harp_xattn.pt found at: {harp_xattn_path}")
        return

    harp_sd = torch.load(harp_xattn_path, map_location="cpu", weights_only=True)
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
    raise ValueError("Could not infer decoder layer path. Please add the correct recursive path for this backbone.")


# ============================================================
# HARP cross-attention modules
# ============================================================
def exists(val):
    return val is not None


def FeedForward(dim: int, mult: int = 4):
    inner = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner, bias=False),
        nn.GELU(),
        nn.Linear(inner, dim, bias=False),
    )


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_memory,
        dim_head=64,
        heads=8,
        only_attend_immediate_memory=True,
        mask_mode="segment",
    ):
        super().__init__()
        assert mask_mode in {"segment", "token"}

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.mask_mode = mask_mode
        self.only_attend_immediate_memory = only_attend_immediate_memory
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
        B, T_txt, _ = x.shape
        _, S, _ = memory.shape
        h = self.heads

        if not use_cached_memory:
            assert exists(placeholder_slot_ids), "placeholder_slot_ids is required unless use_cached_memory=True"

        x = self.norm(x)
        memory = memory.to(dtype=x.dtype)

        q = self.to_q(x)
        k, v = self.to_kv(memory).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        memory_slots = torch.arange(1, S + 1, device=x.device, dtype=torch.long)

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
                dbg["valid_edges"] = int(text_to_memory_mask.sum().item())
                dbg["tokens_with_route"] = int(text_to_memory_mask.any(dim=-1).sum().item())

            dbg["attn_mean"] = float(attn.mean().item())
            dbg["attn_max"] = float(attn.max().item())
            self.last_debug = dbg

        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_memory,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_memory=True,
        mask_mode="segment",
        enable_ff=True,
        attn_gate_init=0.05,
        ff_gate_init=0.05,
    ):
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

    def forward(
        self,
        x,
        memory,
        placeholder_slot_ids=None,
        memory_mask=None,
        use_cached_memory=False,
        xattn_apply_mask=None,
    ):
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
                mask = mask.unsqueeze(-1)
            attn_out = attn_out * mask

        x = x + attn_out * self.attn_gate.tanh()

        if self.ff is not None:
            ff_out = self.ff(x)
            if xattn_apply_mask is not None:
                ff_out = ff_out * mask
            x = x + ff_out * self.ff_gate.tanh()

        return x


def build_placeholder_slot_ids(input_ids, placeholder_token_ids, routing_start_idx=None):
    slot_ids = torch.zeros_like(input_ids, dtype=torch.long)
    for slot_idx, tok_id in enumerate(placeholder_token_ids, start=1):
        slot_ids[input_ids == tok_id] = slot_idx

    if routing_start_idx is not None:
        if routing_start_idx.ndim == 0:
            routing_start_idx = routing_start_idx.unsqueeze(0)
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        valid = pos >= routing_start_idx.unsqueeze(1)
        slot_ids = torch.where(valid, slot_ids, torch.zeros_like(slot_ids))

    return slot_ids


def forward_fill_slot_ids(slot_ids):
    B, T = slot_ids.shape
    pos = torch.arange(T, device=slot_ids.device).unsqueeze(0).expand(B, T)
    seen_pos = torch.where(slot_ids.ne(0), pos, torch.full_like(pos, -1))
    last_pos = torch.cummax(seen_pos, dim=1).values
    gather_pos = last_pos.clamp(min=0)
    active = slot_ids.gather(1, gather_pos)
    active = torch.where(last_pos.ge(0), active, torch.zeros_like(active))
    return active


def last_seen_slot_id(slot_ids):
    B, T = slot_ids.shape
    pos = torch.arange(T, device=slot_ids.device).unsqueeze(0).expand(B, T)
    seen_pos = torch.where(slot_ids.ne(0), pos, torch.full_like(pos, -1))
    last_pos = seen_pos.max(dim=1).values
    gather_pos = last_pos.clamp(min=0).unsqueeze(1)
    last_slot = slot_ids.gather(1, gather_pos)
    last_slot = torch.where(last_pos.ge(0).unsqueeze(1), last_slot, torch.zeros_like(last_slot))
    return last_slot


class HARPLayer(nn.Module):
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

    def forward(self, hidden_states, attention_mask=None, **decoder_layer_kwargs):
        if self.gated_cross_attn_layer is not None:
            if self.harp_x is None or self.harp_mask is None:
                raise ValueError("HARP memory must be conditioned before forward pass")
            if self.placeholder_slot_ids is None:
                raise ValueError("placeholder_slot_ids must be conditioned before forward pass")

            align_module_to_hidden_dtype(self.gated_cross_attn_layer, hidden_states)

            hidden_states = self.gated_cross_attn_layer(
                hidden_states,
                self.harp_x,
                placeholder_slot_ids=self.placeholder_slot_ids,
                memory_mask=self.harp_mask,
                use_cached_memory=bool(self.use_cached_memory),
                xattn_apply_mask=self.xattn_apply_mask,
            )

        return self.decoder_layer(hidden_states, attention_mask=attention_mask, **decoder_layer_kwargs)
    


class HARPLMMixin(nn.Module):
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
                # gated_cross_attn_layer = GatedCrossAttentionBlock(
                #     dim=lang_hidden_size,
                #     dim_memory=mem_hidden_size,
                #     dim_head=xattn_dim_head,
                #     heads=xattn_heads,
                #     ff_mult=xattn_ff_mult,
                #     only_attend_immediate_memory=only_attend_immediate_memory,
                #     mask_mode=mask_mode,
                #     enable_ff=False,
                #     attn_gate_init=0.01,
                #     ff_gate_init=0.0,
                # )

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

    def forward(self, input_ids=None, attention_mask=None, routing_start_idx=None, xattn_apply_mask=None, **kwargs):
        if not getattr(self, "initialized_harp_flamingo", False):
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

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
        return super().forward(**kwargs)


# ============================================================
# Candidate-scoring inference
# ============================================================
@torch.no_grad()
def append_token_ids(input_ids, attention_mask, new_ids: List[int]):
    device = input_ids.device
    new_tensor = torch.tensor([new_ids], dtype=input_ids.dtype, device=device)
    new_attn = torch.ones((1, len(new_ids)), dtype=attention_mask.dtype, device=device)
    input_ids = torch.cat([input_ids, new_tensor], dim=1)
    attention_mask = torch.cat([attention_mask, new_attn], dim=1)
    return input_ids, attention_mask


def truncate_scoring_prefix(
    prefix_ids: List[int],
    max_prefix_tokens: int,
    keep_head_tokens: int,
) -> List[int]:
    if max_prefix_tokens <= 0 or len(prefix_ids) <= max_prefix_tokens:
        return prefix_ids

    keep_head = min(max(0, keep_head_tokens), max_prefix_tokens - 1)
    keep_tail = max_prefix_tokens - keep_head

    if keep_head <= 0:
        return prefix_ids[-max_prefix_tokens:]

    return prefix_ids[:keep_head] + prefix_ids[-keep_tail:]


def truncate_scoring_prefix_preserve_target(
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
        route_start = int(routing_start_idx[0].item()) if routing_start_idx is not None else base_len
        xmask[:, route_start:] = 1.0

        model_inputs["routing_start_idx"] = routing_start_idx
        model_inputs["xattn_apply_mask"] = xmask

    outputs = model(**model_inputs)
    cand_logits = outputs.logits[:, base_len - 1 : base_len - 1 + cand_len, :].float()
    target = torch.tensor(cand_ids, dtype=torch.long, device=device).unsqueeze(0)

    token_logprobs = F.log_softmax(cand_logits, dim=-1)
    token_logprobs = token_logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1).squeeze(0)

    return {
        "sum_logprob": float(token_logprobs.sum().item()),
        "mean_logprob": float(token_logprobs.mean().item()),
        "token_count": int(cand_len),
    }


def select_candidate_from_scored(
    scored: List[Dict[str, Any]],
    *,
    decode_mode: str,
    sample_temperature: float,
    sample_top_p: float,
    sample_top_k: int,
    sample_generator: Optional[torch.Generator],
):
    """
    scored: list of dicts with keys:
        rhs, score, mean_logprob, sum_logprob, token_count
    Returns:
        chosen_candidate_dict, ordered_scored_list
    """
    ordered = sorted(scored, key=lambda x: (x["score"], x["sum_logprob"]), reverse=True)

    if len(ordered) == 1 or decode_mode == "greedy":
        chosen = copy.deepcopy(ordered[0])
        chosen["sample_prob"] = 1.0
        chosen["sample_rank"] = 1
        return chosen, ordered

    logits = torch.tensor([x["score"] for x in ordered], dtype=torch.float32)

    temp = max(float(sample_temperature), 1e-6)
    logits = logits / temp

    # top-k filter
    if sample_top_k > 0 and sample_top_k < logits.numel():
        topk_vals, topk_idx = torch.topk(logits, k=sample_top_k)
        masked = torch.full_like(logits, -float("inf"))
        masked[topk_idx] = logits[topk_idx]
        logits = masked

    # top-p / nucleus filter
    if sample_top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=0)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        keep = cumulative_probs <= float(sample_top_p)
        if keep.numel() > 0:
            keep[0] = True  # always keep the best one

        masked = torch.full_like(logits, -float("inf"))
        kept_idx = sorted_idx[keep]
        masked[kept_idx] = logits[kept_idx]
        logits = masked

    probs = torch.softmax(logits, dim=0)

    if (not torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0:
        probs = torch.zeros_like(logits)
        probs[0] = 1.0

    chosen_idx = int(torch.multinomial(probs, num_samples=1, generator=sample_generator).item())

    probs_list = probs.tolist()
    for i, p in enumerate(probs_list):
        ordered[i]["sample_prob"] = float(p)

    chosen = copy.deepcopy(ordered[chosen_idx])
    chosen["sample_prob"] = float(probs[chosen_idx].item())
    chosen["sample_rank"] = int(chosen_idx + 1)
    return chosen, ordered


def annotate_candidate_uniqueness(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Marks candidates in-place with:
      - is_unique
      - duplicate_of_sample_id
      - score_rank_among_unique
    Returns the unique candidates sorted by model score.
    """
    seen = {}
    unique = []

    for cand in candidates:
        key = cand["canonical_prediction"]
        if key not in seen:
            cand["is_unique"] = True
            cand["duplicate_of_sample_id"] = None
            seen[key] = cand["sample_id"]
            unique.append(cand)
        else:
            cand["is_unique"] = False
            cand["duplicate_of_sample_id"] = seen[key]

    unique_sorted = sorted(
        unique,
        key=lambda x: (x["sequence_score"], x["sequence_sum_logprob"]),
        reverse=True,
    )

    for rank, cand in enumerate(unique_sorted, start=1):
        cand["score_rank_among_unique"] = rank

    for cand in candidates:
        if "score_rank_among_unique" not in cand:
            cand["score_rank_among_unique"] = None

    return unique_sorted


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
    debug_topk: int = 0,
    candidate_max_prefix_tokens: int = 0,
    candidate_keep_head_tokens: int = 0,
    decode_mode: str = "greedy",              # NEW
    sample_temperature: float = 1.0,          # NEW
    sample_top_p: float = 1.0,                # NEW
    sample_top_k: int = 0,                    # NEW
    sample_generator: Optional[torch.Generator] = None,  # NEW
):
    assert score_reduction in {"mean", "sum"}
    assert decode_mode in {"greedy", "sample"}

    device = get_first_real_device(model)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    if routing_start_idx is None:
        routing_start_idx = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)

    parts = []
    current_label = None
    site_debug = []

    sequence_score = 0.0
    sequence_sum_logprob = 0.0
    site_count = 0

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

            base_prefix_ids = input_ids[0].tolist()
            effective_route_idx = int(routing_start_idx.item()) if routing_start_idx is not None else None

            truncated_prefix_ids, effective_route_idx = truncate_scoring_prefix_preserve_target(
                prefix_ids=base_prefix_ids,
                routing_start_idx=effective_route_idx,
                max_prefix_tokens=candidate_max_prefix_tokens,
                keep_head_tokens=candidate_keep_head_tokens,
            )

            base_input = torch.tensor([truncated_prefix_ids], dtype=torch.long, device=device)
            base_mask = torch.ones_like(base_input)

            effective_routing_start_idx = None
            if effective_route_idx is not None:
                effective_routing_start_idx = torch.tensor(
                    [effective_route_idx],
                    dtype=torch.long,
                    device=device,
                )

            for rhs in candidates:
                stats = score_rhs_candidate_suffix(
                    model=model,
                    tok=tok,
                    base_input_ids=base_input,
                    base_attention_mask=base_mask,
                    candidate_text=rhs + "\n",
                    routing_start_idx=effective_routing_start_idx,
                    use_harp=use_harp,
                )
                scored.append(
                    {
                        "rhs": rhs,
                        "score": stats["mean_logprob"] if score_reduction == "mean" else stats["sum_logprob"],
                        "mean_logprob": stats["mean_logprob"],
                        "sum_logprob": stats["sum_logprob"],
                        "token_count": stats["token_count"],
                    }
                )

            chosen, ordered = select_candidate_from_scored(
                scored,
                decode_mode=decode_mode,
                sample_temperature=sample_temperature,
                sample_top_p=sample_top_p,
                sample_top_k=sample_top_k,
                sample_generator=sample_generator,
            )

            chosen_text = chosen["rhs"] + "\n"
            chosen_ids = tok(chosen_text, add_special_tokens=False)["input_ids"]
            input_ids, attention_mask = append_token_ids(input_ids, attention_mask, chosen_ids)
            parts.append(chosen_text)

            sequence_score += float(chosen["score"])
            sequence_sum_logprob += float(chosen["sum_logprob"])
            site_count += 1

            if debug_topk > 0:
                site_debug.append(
                    {
                        "label": label,
                        "lhs": lhs,
                        "chosen_rhs": chosen["rhs"],
                        "chosen_score": float(chosen["score"]),
                        "chosen_sum_logprob": float(chosen["sum_logprob"]),
                        "chosen_sample_prob": float(chosen.get("sample_prob", 1.0)),
                        "chosen_sample_rank": int(chosen.get("sample_rank", 1)),
                        "top_candidates": ordered[:debug_topk],
                    }
                )

        prediction = "".join(parts).rstrip()
        canonical_prediction = canonicalize_generation(prediction)

        return {
            "prediction": prediction,
            "canonical_prediction": canonical_prediction,
            "site_debug": site_debug,
            "sequence_score": float(sequence_score),
            "sequence_sum_logprob": float(sequence_sum_logprob),
            "sequence_mean_site_score": float(sequence_score / max(site_count, 1)),
            "site_count": int(site_count),
            "decode_mode": decode_mode,
        }

    finally:
        if hasattr(model, "clear_harp"):
            model.clear_harp()


# ============================================================
# Metrics
# ============================================================
def evaluate_prediction(reference_target: str, raw_generation: str) -> Dict[str, object]:
    pred_text = canonicalize_generation(raw_generation)
    ref_text = reference_target.strip()

    ref_assign = parse_assignment_dict(ref_text)
    pred_assign = parse_assignment_dict(pred_text)
    expected_keys = list(ref_assign.keys())

    exact_value_match_count = sum(
        (k in pred_assign) and (pred_assign[k] == ref_assign[k]) for k in expected_keys
    )

    return {
        "canonical_prediction": pred_text,
        "value_accuracy_over_expected": exact_value_match_count / max(len(expected_keys), 1),
        "n_expected": len(expected_keys),
        "n_predicted": len(pred_assign),
    }


# ============================================================
# Model loading
# ============================================================
def build_tokenizer(tokenizer_source: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    special_tokens = list(OBJ_TOKENS.values()) + SOURCE_PLACEHOLDER_TOKENS + TARGET_PLACEHOLDER_TOKENS
    tok.add_special_tokens({"additional_special_tokens": special_tokens})

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


# def load_base_model(model_name: str, use_4bit: bool = True, device_map: str = "auto"):
#     quantization_config = None
#     if use_4bit:
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=quantization_config,
#         torch_dtype=(torch.bfloat16 if use_4bit and torch.cuda.is_available() else None),
#         device_map=device_map,
#         trust_remote_code=True,
#     )

#     model.config.use_cache = False
#     return model


def load_base_model(
    model_name: str,
    use_4bit: bool = True,
    device_map: str = "auto",
):
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if device_map == "auto":
        device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    return model


def attach_harp_modules(
    model,
    tok,
    mem_dim: int,
    every_n_layers: int,
    xattn_heads: int,
    xattn_dim_head: int,
    xattn_ff_mult: int,
):
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
        mem_hidden_size=mem_dim,
        cross_attn_every_n_layers=every_n_layers,
        gradient_checkpointing=False,
        xattn_heads=xattn_heads,
        xattn_dim_head=xattn_dim_head,
        xattn_ff_mult=xattn_ff_mult,
        only_attend_immediate_memory=True,
        mask_mode="segment",
    )

    print(f"[HARP-XATTN] decoder_layers_attr_name={decoder_layers_attr_name}")
    print(f"[HARP-XATTN] inserted gated xattn every {every_n_layers} decoder layers")
    move_harp_modules_to_model_device(model)


def load_stage_model(args, tok):
    base = load_base_model(
        model_name=args.model,
        use_4bit=not args.no_4bit,
        device_map=args.device_map,
    )

    base.resize_token_embeddings(len(tok))
    if hasattr(base.config, "tie_word_embeddings"):
        base.config.tie_word_embeddings = True
    try:
        base.tie_weights()
    except Exception:
        pass

    lora_adapter_dir = args.lora_adapter_dir or args.adapter_dir
    if not lora_adapter_dir:
        raise ValueError("--adapter_dir (or --lora_adapter_dir) is required for stage1/stage2 inference")

    model = load_peft_adapter_strict(base, lora_adapter_dir)

    if args.stage in {"stage2", "stage3"}:
        attach_harp_modules(
            model,
            tok,
            mem_dim=args.mem_dim,
            every_n_layers=args.every_n_layers,
            xattn_heads=args.xattn_heads,
            xattn_dim_head=args.xattn_dim_head,
            xattn_ff_mult=args.xattn_ff_mult,
        )

        harp_xattn_path = args.harp_xattn_path
        if not harp_xattn_path:
            harp_xattn_path = os.path.join(args.adapter_dir, "harp_xattn.pt")

        load_partial_harp_xattn(
            model,
            harp_xattn_path,
            tag=f"HARP-LOAD-{args.stage.upper()}",
        )

    model.eval()
    return model



def has_adapter_weights(adapter_dir: str) -> bool:
    return any(
        os.path.isfile(os.path.join(adapter_dir, fn))
        for fn in ("adapter_model.safetensors", "adapter_model.bin")
    )


def load_peft_adapter_strict(base, adapter_dir: str):
    if not adapter_dir:
        raise ValueError("Empty adapter_dir")

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    if not has_adapter_weights(adapter_dir):
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.bin found in: {adapter_dir}"
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)

    missing_adapter_warnings = [
        str(w.message) for w in caught
        if "Found missing adapter keys" in str(w.message)
    ]

    if missing_adapter_warnings:
        raise RuntimeError(
            f"LoRA adapter did not load cleanly from {adapter_dir}.\n"
            "For stage2 inference, use the stage1 LoRA adapter directory as --lora_adapter_dir "
            "(or --adapter_dir), and load stage2 HARP weights via --harp_xattn_path."
        )

    print(f"[ADAPTER] loaded clean PEFT adapter from: {adapter_dir}")
    return model


# ============================================================
# Case prediction
# ============================================================
def predict_case(
    *,
    model,
    tok,
    case: InferenceCase,
    rhs_candidate_bank: Dict[str, List[str]],
    stage: str,
    max_prompt_tokens: int,
    mem_bank: Optional[Dict[str, dict]],
    mem_dim: int,
    max_slots: int,
    score_reduction: str,
    debug_topk: int,
    candidate_max_prefix_tokens: int,
    candidate_keep_head_tokens: int,
    print_xattn_debug_flag: bool,
    num_samples: int,                 # NEW
    no_greedy_first: bool,            # NEW
    sample_temperature: float,        # NEW
    sample_top_p: float,              # NEW
    sample_top_k: int,                # NEW
    sample_seed: int,                 # NEW
) -> Dict[str, Any]:
    prompt = build_prompt_for_case(
        code=case.source_text,
        obj_mode=case.obj_mode,
        w_lat=case.w_lat,
        w_area=case.w_area,
    )

    enc = tok(prompt, add_special_tokens=False)
    prompt_ids = enc["input_ids"][-max_prompt_tokens:] if len(enc["input_ids"]) > max_prompt_tokens else enc["input_ids"]

    device = get_first_real_device(model)
    routing_start_idx = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)

    harp_x = None
    harp_mask = None
    if stage in {"stage2", "stage3"}:
        if mem_bank is None:
            raise ValueError("mem_bank must be provided for stage2/stage3 inference")
        harp_x, harp_mask = get_real_memory_pack_for_kernel(
            mem_bank,
            case.kernel_name,
            max_slots=max_slots,
            mem_dim=mem_dim,
        )

    decode_plan = []
    if not no_greedy_first:
        decode_plan.append(("greedy", None))

    while len(decode_plan) < num_samples:
        per_sample_seed = int(sample_seed + len(decode_plan))
        decode_plan.append(("sample", per_sample_seed))

    candidates = []

    for sample_id, (decode_mode, this_seed) in enumerate(decode_plan):
        sample_generator = None
        if decode_mode == "sample":
            sample_generator = torch.Generator(device="cpu")
            sample_generator.manual_seed(int(this_seed))

        out = constrained_decode_rhs_by_candidate_scoring(
            model=model,
            tok=tok,
            prompt_ids=prompt_ids,
            source_text=case.source_text,
            rhs_candidate_bank=rhs_candidate_bank,
            score_reduction=score_reduction,
            harp_x=harp_x,
            harp_mask=harp_mask,
            routing_start_idx=routing_start_idx,
            debug_topk=debug_topk,
            candidate_max_prefix_tokens=candidate_max_prefix_tokens,
            candidate_keep_head_tokens=candidate_keep_head_tokens,
            decode_mode=decode_mode,
            sample_temperature=sample_temperature,
            sample_top_p=sample_top_p,
            sample_top_k=sample_top_k,
            sample_generator=sample_generator,
        )

        cand = {
            "sample_id": int(sample_id),
            "decode_mode": decode_mode,
            "sample_seed": this_seed,
            "prediction": out["canonical_prediction"],
            "canonical_prediction": out["canonical_prediction"],
            "sequence_score": float(out["sequence_score"]),
            "sequence_sum_logprob": float(out["sequence_sum_logprob"]),
            "sequence_mean_site_score": float(out["sequence_mean_site_score"]),
            "site_count": int(out["site_count"]),
            "site_debug": out["site_debug"],
        }

        if case.reference_target is not None:
            metrics = evaluate_prediction(case.reference_target, out["canonical_prediction"])
            cand.update(metrics)

        candidates.append(cand)

        if print_xattn_debug_flag and stage in {"stage2", "stage3"} and sample_id == 0:
            print_xattn_forward_stats(model)

    unique_candidates_sorted = annotate_candidate_uniqueness(candidates)

    row = {
        "kernel_name": case.kernel_name,
        "obj_mode": case.obj_mode,
        "w_lat": case.w_lat,
        "w_area": case.w_area,
        "prompt_token_count": len(prompt_ids),
        "n_generated": len(candidates),
        "n_unique": len(unique_candidates_sorted),
        "best_unique_sample_id_by_model_score": (
            int(unique_candidates_sorted[0]["sample_id"]) if unique_candidates_sorted else None
        ),
        "unique_sample_ids_by_model_score": [int(x["sample_id"]) for x in unique_candidates_sorted],
        "candidates": candidates,
    }

    if stage in {"stage2", "stage3"} and harp_mask is not None:
        row["memory_active_slots"] = int(harp_mask.sum().item())

    if case.reference_target is not None:
        row["reference_target"] = case.reference_target

        unique_with_ref = [x for x in unique_candidates_sorted if "value_accuracy_over_expected" in x]
        if unique_with_ref:
            best_acc = sorted(
                unique_with_ref,
                key=lambda x: (x["value_accuracy_over_expected"], x["sequence_score"]),
                reverse=True,
            )[0]
            row["best_unique_sample_id_by_value_accuracy"] = int(best_acc["sample_id"])
            row["best_value_accuracy_over_expected"] = float(best_acc["value_accuracy_over_expected"])

    return row


# ============================================================
# CLI utilities
# ============================================================
def build_single_case_from_args(args) -> InferenceCase:
    if not args.kernel_name:
        raise ValueError("--kernel_name is required for single-case inference")

    if not args.code_file:
        raise ValueError("--code_file is required for single-case inference")

    with open(args.code_file, "r", encoding="utf-8") as f:
        source_text = f.read()

    if args.objective:
        obj = args.objective.strip().lower()

        if obj in {"pareto_latency_extreme", "latency_extreme", "latency", "min_lat", "min_latency"}:
            obj_mode, w_lat, w_area = "PARETO_LATENCY_EXTREME", 1.0, 0.0
        elif obj in {"pareto_area_extreme", "area_extreme", "area", "min_area"}:
            obj_mode, w_lat, w_area = "PARETO_AREA_EXTREME", 0.0, 1.0
        elif obj in {"pareto_knee", "knee", "balanced", "balance"}:
            obj_mode, w_lat, w_area = "PARETO_KNEE", 0.5, 0.5
        else:
            raise ValueError(f"Unknown objective: {args.objective}")
    else:
        w_lat, w_area = normalize_weight_pair(args.w_lat, args.w_area)
        obj_mode = mode_from_weights(w_lat, w_area)

    return InferenceCase(
        kernel_name=args.kernel_name,
        source_text=source_text,
        obj_mode=obj_mode,
        w_lat=w_lat,
        w_area=w_area,
        reference_target=None,
    )


def load_cases(args) -> List[InferenceCase]:
    if args.input_jsonl:
        return load_inference_cases_jsonl(args.input_jsonl)
    return [build_single_case_from_args(args)]


def build_or_load_rhs_bank(args) -> Dict[str, List[str]]:
    if args.rhs_candidate_bank_json:
        bank = load_rhs_candidate_bank(args.rhs_candidate_bank_json)
        print("[BANK] Loaded RHS candidate bank from JSON")
        return bank

    if not args.candidate_bank_dataset:
        raise ValueError("Provide either --rhs_candidate_bank_json or --candidate_bank_dataset")

    rows = load_rows(args.candidate_bank_dataset)
    rows = maybe_filter_rows_for_candidate_bank(rows, args.candidate_bank_exclude_families)
    bank = build_rhs_candidate_bank(rows)
    print("[BANK] Built RHS candidate bank from dataset")

    if args.save_rhs_candidate_bank_json:
        save_rhs_candidate_bank(args.save_rhs_candidate_bank_json, bank)
        print(f"[BANK] Saved RHS candidate bank -> {args.save_rhs_candidate_bank_json}")

    return bank


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    # model / stage
    ap.add_argument("--stage", type=str, required=True, choices=["stage1", "stage2", "stage3"])
    ap.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-7b-base")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--lora_adapter_dir", type=str, default="")
    ap.add_argument("--harp_xattn_path", type=str, default="")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")

    # candidate bank
    ap.add_argument("--candidate_bank_dataset", type=str, default="")
    ap.add_argument("--candidate_bank_exclude_families", type=str, default="")
    ap.add_argument("--rhs_candidate_bank_json", type=str, default="")
    ap.add_argument("--save_rhs_candidate_bank_json", type=str, default="")

    # stage2 memory
    ap.add_argument("--memory_dir", type=str, default="")
    ap.add_argument("--mem_dim", type=int, default=32)
    ap.add_argument("--max_slots", type=int, default=64)
    ap.add_argument("--every_n_layers", type=int, default=16)
    ap.add_argument("--xattn_heads", type=int, default=4)
    ap.add_argument("--xattn_dim_head", type=int, default=64)
    ap.add_argument("--xattn_ff_mult", type=int, default=1)

    # input cases
    ap.add_argument("--input_jsonl", type=str, default="")
    ap.add_argument("--kernel_name", type=str, default="")
    ap.add_argument("--code_file", type=str, default="")
    ap.add_argument("--objective", type=str, default="")
    ap.add_argument("--w_lat", type=float, default=0.5)
    ap.add_argument("--w_area", type=float, default=0.5)

    # decoding / scoring
    ap.add_argument("--max_prompt_tokens", type=int, default=7168)
    ap.add_argument("--score_reduction", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--debug_topk", type=int, default=0)
    ap.add_argument("--candidate_max_prefix_tokens", type=int, default=0)
    ap.add_argument("--candidate_keep_head_tokens", type=int, default=256)
    ap.add_argument("--print_xattn_debug", action="store_true")

    # output
    ap.add_argument("--output_jsonl", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    ap.add_argument("--print_predictions", action="store_true")

    # samples
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--no_greedy_first", action="store_true")
    ap.add_argument("--sample_temperature", type=float, default=0.8)
    ap.add_argument("--sample_top_p", type=float, default=1.0)
    ap.add_argument("--sample_top_k", type=int, default=0)
    ap.add_argument("--sample_seed", type=int, default=123)

    args = ap.parse_args()

    if args.stage in {"stage2", "stage3"} and not args.memory_dir:
        raise ValueError("--memory_dir is required for stage2/stage3 inference")

    rhs_candidate_bank = build_or_load_rhs_bank(args)
    tok = build_tokenizer(args.model)
    model = load_stage_model(args, tok)
    mem_bank = load_memory_bank(args.memory_dir) if args.stage in {"stage2", "stage3"} else None
    cases = load_cases(args)

    for n, p in model.named_parameters():
        if n.endswith("attn_gate"):
            print(n, float(p.item()), float(p.tanh().item()))

    outputs = []
    for idx, case in enumerate(cases, start=1):
        print(f"[CASE {idx}/{len(cases)}] kernel={case.kernel_name} obj={case.obj_mode}")
        row = predict_case(
            model=model,
            tok=tok,
            case=case,
            rhs_candidate_bank=rhs_candidate_bank,
            stage=args.stage,
            max_prompt_tokens=args.max_prompt_tokens,
            mem_bank=mem_bank,
            mem_dim=args.mem_dim,
            max_slots=args.max_slots,
            score_reduction=args.score_reduction,
            debug_topk=args.debug_topk,
            candidate_max_prefix_tokens=args.candidate_max_prefix_tokens,
            candidate_keep_head_tokens=args.candidate_keep_head_tokens,
            print_xattn_debug_flag=args.print_xattn_debug,
            num_samples=args.num_samples,
            no_greedy_first=args.no_greedy_first,
            sample_temperature=args.sample_temperature,
            sample_top_p=args.sample_top_p,
            sample_top_k=args.sample_top_k,
            sample_seed=args.sample_seed,
        )
        outputs.append(row)

        if args.print_predictions:
            print("-" * 100)
            print(
                f"[CASE SUMMARY] generated={row['n_generated']} "
                f"unique={row['n_unique']} "
                f"best_unique_sample_id_by_model_score={row['best_unique_sample_id_by_model_score']}"
            )
            for cand in row["candidates"]:
                print(
                    f"[SAMPLE {cand['sample_id']}] "
                    f"mode={cand['decode_mode']} "
                    f"is_unique={cand['is_unique']} "
                    f"dup_of={cand['duplicate_of_sample_id']} "
                    f"seq_score={cand['sequence_score']:.6f} "
                    f"seq_sum_logprob={cand['sequence_sum_logprob']:.6f}"
                )
                if "value_accuracy_over_expected" in cand:
                    print(f"  [ACC] value_accuracy_over_expected={cand['value_accuracy_over_expected']:.6f}")
                print(cand["canonical_prediction"])
                print("-" * 60)
            print("-" * 100)

    if args.output_jsonl:
        dump_jsonl(args.output_jsonl, outputs)
        print(f"[DONE] wrote JSONL -> {args.output_jsonl}")

    if args.output_json:
        payload = outputs[0] if len(outputs) == 1 else outputs
        dump_json(args.output_json, payload)
        print(f"[DONE] wrote JSON -> {args.output_json}")

    if not args.output_jsonl and not args.output_json:
        payload = outputs[0] if len(outputs) == 1 else outputs
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
