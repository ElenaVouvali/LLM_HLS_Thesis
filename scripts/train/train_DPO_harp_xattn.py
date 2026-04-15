import argparse
import re
import gc
import importlib.util
import json
import math
import os
import random
import numpy as np

from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_pt_utils import LengthGroupedSampler
from peft import PeftModel, prepare_model_for_kbit_training



def build_prompt(mod, source_text: str, obj_mode: str) -> str:
    return mod.build_prompt(source_text, obj_mode)

def import_module_from_path(module_path: str, module_name: str = "sft_mod"):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def build_selected_splits(mod, args, rows):
    if args.split_json:
        split_spec = mod.load_split_spec(args.split_json)
        raw_train_rows, raw_val_rows, raw_test_rows = mod.apply_split_spec(rows, split_spec)
        print(f"[INFO] Loaded split from {args.split_json}")

    elif args.split_mode == "family":
        val_fams = {mod.normalize_name(x) for x in args.val_families.split(";") if x.strip()}
        test_fams = {mod.normalize_name(x) for x in args.test_families.split(";") if x.strip()}
        raw_train_rows, raw_val_rows, raw_test_rows = mod.split_by_family(rows, val_fams, test_fams)
        print("[INFO] val_families:", sorted(val_fams))
        print("[INFO] test_families:", sorted(test_fams))

    else:
        raw_train_rows, raw_val_rows, raw_test_rows = mod.split_rows_random_design(
            rows,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
            stratify_by_kernel=args.stratify_by_kernel,
        )
        print(
            f"[INFO] random design-point split with "
            f"val_ratio={args.val_ratio}, test_ratio={args.test_ratio}, "
            f"split_seed={args.split_seed}, stratify_by_kernel={args.stratify_by_kernel}"
        )

    if args.save_split_json:
        mod.save_split_spec(args.save_split_json, raw_train_rows, raw_val_rows, raw_test_rows)
        print(f"[INFO] Saved split spec -> {args.save_split_json}")

    train_rows, _ = mod.select_goal_rows(
        raw_train_rows,
        goal_mode=args.objective,
        top_k=args.top_k,
        domination_penalty=args.goal_domination_penalty,
        max_dominated_gap=args.goal_max_dominated_gap,
        score_weight_min=args.score_weight_min,
        score_weight_power=args.score_weight_power,
    )

    val_rows, _ = mod.select_goal_rows(
        raw_val_rows,
        goal_mode=args.objective,
        top_k=args.top_k,
        domination_penalty=args.goal_domination_penalty,
        max_dominated_gap=args.goal_max_dominated_gap,
        score_weight_min=args.score_weight_min,
        score_weight_power=args.score_weight_power,
    )

    test_rows, _ = mod.select_goal_rows(
        raw_test_rows,
        goal_mode=args.objective,
        top_k=args.top_k,
        domination_penalty=args.goal_domination_penalty,
        max_dominated_gap=args.goal_max_dominated_gap,
        score_weight_min=args.score_weight_min,
        score_weight_power=args.score_weight_power,
    )

    print(f"[INFO] Selected split sizes: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    return train_rows, val_rows, test_rows


def dump_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


ASSIGN_RE = re.compile(
    r"^(auto\{_[A-Z0-9]+(?:_[A-Z0-9]+)*_L\d+\})\s*=\s*(.+)$",
    re.IGNORECASE,
)

def parse_target_map(target_text: str) -> Dict[str, str]:
    """
    Parse:
        auto{_PIPE_L3} = 1
    into:
        {"auto{_PIPE_L3}": "1"}
    """
    out = {}
    for raw_line in target_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = ASSIGN_RE.match(line)
        if m is None:
            continue
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        out[lhs] = rhs
    return out


def relative_improvement(old_val: float, new_val: float) -> float:
    """
    Positive means 'new_val' is better (smaller) than 'old_val'.
    """
    old_val = float(old_val)
    new_val = float(new_val)
    denom = max(abs(old_val), 1e-12)
    return (old_val - new_val) / denom


class GoalPreferencePairBuilder:
    def __init__(
        self,
        mod,
        objective: str,
        chosen_top_k: int = 3,
        hard_window: int = 8,
        hard_negatives_per_chosen: int = 2,
        medium_negatives_per_chosen: int = 1,
        min_score_gap: float = 0.02,
        hard_gap_max: float = 0.15,
        medium_gap_max: float = 0.35,
        min_primary_rel_gain: float = 0.02,
        min_edit_distance: int = 1,
        min_edit_frac: float = 0.0,
        max_edit_frac: float = 1.0,
        min_supervised_sites: int = 2,
        min_site_coverage: float = 0.85,
        require_same_supervised_schema: bool = True,
        balanced_min_sum_gain: float = 0.02,
        balanced_max_axis_loss: float = 0.25,
        balanced_min_better_axis_gain: float = 0.03,
    ):
        self.mod = mod
        self.objective = objective
        self.chosen_top_k = chosen_top_k
        self.hard_window = hard_window
        self.hard_negatives_per_chosen = hard_negatives_per_chosen
        self.medium_negatives_per_chosen = medium_negatives_per_chosen
        self.min_score_gap = float(min_score_gap)
        self.hard_gap_max = float(hard_gap_max)
        self.medium_gap_max = float(medium_gap_max)
        self.min_primary_rel_gain = float(min_primary_rel_gain)
        self.min_edit_distance = int(min_edit_distance)
        self.min_edit_frac = float(min_edit_frac)
        self.max_edit_frac = float(max_edit_frac)
        self.min_supervised_sites = int(min_supervised_sites)
        self.min_site_coverage = float(min_site_coverage)
        self.require_same_supervised_schema = bool(require_same_supervised_schema)
        self.balanced_min_sum_gain = float(balanced_min_sum_gain)
        self.balanced_max_axis_loss = float(balanced_max_axis_loss)
        self.balanced_min_better_axis_gain = float(balanced_min_better_axis_gain)

    def _canonical_row(self, row: dict):
        try:
            completion, meta = self.mod.build_partial_deterministic_target_text(
                row["input"],
                row["target"],
                min_supervised_sites=self.min_supervised_sites,
            )
        except ValueError:
            return None

        if meta["coverage"] < self.min_site_coverage:
            return None

        rhs_map = parse_target_map(completion)
        schema_key = tuple(
            lhs for _, lhs in self.mod.extract_ordered_lhs_plan(row["input"])
            if lhs in rhs_map
        )

        return {
            "row": row,
            "completion": completion,
            "rhs_map": rhs_map,
            "schema_key": schema_key,
            "score": float(row.get("_score", 1e9)),
            "rank": int(row.get("_rank_within_kernel", 10**9)),
            "num_sites": int(meta["n_supervised"]),
        }

    def _directive_diff(self, a: dict, b: dict):
        if a["schema_key"] != b["schema_key"]:
            return 0, 0.0
        keys = list(a["schema_key"])
        diff = sum(a["rhs_map"].get(k) != b["rhs_map"].get(k) for k in keys)
        frac = diff / max(1, len(keys))
        return int(diff), float(frac)

    def _rel_gains(self, chosen: dict, rejected: dict):
        ch_lat = float(chosen["row"]["latency"])
        rj_lat = float(rejected["row"]["latency"])
        ch_area = float(chosen["row"]["area"])
        rj_area = float(rejected["row"]["area"])

        lat_gain = (rj_lat - ch_lat) / max(abs(rj_lat), 1e-12)
        area_gain = (rj_area - ch_area) / max(abs(rj_area), 1e-12)
        return float(lat_gain), float(area_gain)

    def _primary_gain_ok(self, lat_gain: float, area_gain: float):
        if self.objective == "PARETO_LATENCY_EXTREME":
            return lat_gain >= self.min_primary_rel_gain
        if self.objective == "PARETO_AREA_EXTREME":
            return area_gain >= self.min_primary_rel_gain

        better_axis = max(lat_gain, area_gain)
        worse_axis = min(lat_gain, area_gain)
        net_gain = lat_gain + area_gain

        return (
            better_axis >= self.balanced_min_better_axis_gain
            and worse_axis >= -self.balanced_max_axis_loss
            and net_gain >= self.balanced_min_sum_gain
        )

    def build(self, rows: List[dict]) -> List[dict]:
        by_kernel = defaultdict(list)
        for row in rows:
            by_kernel[row["kernel_name"]].append(row)

        pairs = []

        for kernel_name, kernel_rows in by_kernel.items():
            ranked_rows = sorted(
                kernel_rows,
                key=lambda r: (int(r.get("_rank_within_kernel", 10**9)), float(r.get("_score", 1e9))),
            )

            uniq = []
            seen = set()
            for row in ranked_rows:
                rec = self._canonical_row(row)
                if rec is None:
                    continue
                if rec["completion"] in seen:
                    continue
                seen.add(rec["completion"])
                uniq.append(rec)

            if len(uniq) < 4:
                continue

            chosen_pool = uniq[: min(self.chosen_top_k, len(uniq))]

            for chosen_idx, chosen in enumerate(chosen_pool):
                hard_pool = []
                medium_pool = []

                for rejected in uniq[chosen_idx + 1:]:
                    gap = float(rejected["score"] - chosen["score"])
                    if gap > self.medium_gap_max:
                        break

                    if self.require_same_supervised_schema and chosen["schema_key"] != rejected["schema_key"]:
                        continue

                    lat_gain, area_gain = self._rel_gains(chosen, rejected)
                    if not self._primary_gain_ok(lat_gain, area_gain):
                        continue
                    if gap < self.min_score_gap:
                        continue

                    diff_count, diff_frac = self._directive_diff(chosen, rejected)
                    if diff_count < self.min_edit_distance:
                        continue
                    if diff_frac < self.min_edit_frac or diff_frac > self.max_edit_frac:
                        continue

                    rec = {
                        "kernel_name": kernel_name,
                        "family": chosen["row"].get("_family"),
                        "source_text": chosen["row"]["input"],
                        "obj_mode": self.objective,
                        "prompt": self.mod.build_prompt(chosen["row"]["input"], self.objective),
                        "chosen": chosen["completion"],
                        "rejected": rejected["completion"],
                        "chosen_score": float(chosen["score"]),
                        "rejected_score": float(rejected["score"]),
                        "score_gap": float(gap),
                        "directive_diff_count": int(diff_count),
                        "directive_diff_frac": float(diff_frac),
                        "latency_rel_gain": float(lat_gain),
                        "area_rel_gain": float(area_gain),
                        "chosen_rank": int(chosen["rank"]),
                        "rejected_rank": int(rejected["rank"]),
                        "num_sites": int(chosen["num_sites"]),
                        "pair_tier": "hard" if (rejected["rank"] - chosen["rank"]) <= self.hard_window else "medium",
                    }

                    if rec["pair_tier"] == "hard" and gap <= self.hard_gap_max:
                        hard_pool.append(rec)
                    else:
                        medium_pool.append(rec)

                selected = (
                    hard_pool[: self.hard_negatives_per_chosen]
                    + medium_pool[: self.medium_negatives_per_chosen]
                )

                if not selected:
                    fallback = hard_pool if hard_pool else medium_pool
                    if fallback:
                        selected = [fallback[0]]

                pairs.extend(selected)

        dedup = []
        seen = set()
        for p in pairs:
            key = (p["kernel_name"], p["obj_mode"], p["chosen"], p["rejected"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(p)
        return dedup

    

def _q(vals, q):
    if not vals:
        return None
    return float(np.quantile(np.array(vals, dtype=np.float64), q))


def classify_knee_pair(p):
    lat = float(p["latency_rel_gain"])
    area = float(p["area_rel_gain"])

    if lat >= 0.0 and area >= 0.0:
        return "balanced"
    if area > 0.0 and lat < 0.0:
        return "area_favoring"
    if lat > 0.0 and area < 0.0:
        return "latency_favoring"
    return "other"


def rebalance_knee_pairs(rows, seed=123, max_ratio=1.25):
    rng = random.Random(seed)

    balanced = [r for r in rows if classify_knee_pair(r) == "balanced"]
    area_fav = [r for r in rows if classify_knee_pair(r) == "area_favoring"]
    lat_fav = [r for r in rows if classify_knee_pair(r) == "latency_favoring"]
    other = [r for r in rows if classify_knee_pair(r) == "other"]

    rng.shuffle(area_fav)
    rng.shuffle(lat_fav)

    if len(lat_fav) > 0:
        area_cap = int(max_ratio * len(lat_fav))
        lat_cap = int(max_ratio * len(area_fav))
        area_fav = area_fav[:area_cap]
        lat_fav = lat_fav[:lat_cap]

    out = balanced + area_fav + lat_fav + other
    rng.shuffle(out)
    return out


def audit_preference_pairs(name: str, rows: List[dict]) -> None:
    print(f"\n[PAIR-AUDIT] {name}")
    if not rows:
        print("  no pairs")
        return

    by_obj = Counter(r["obj_mode"] for r in rows)
    by_kernel_obj = Counter((r["kernel_name"], r["obj_mode"]) for r in rows)
    by_tier = Counter(r.get("pair_tier", "unknown") for r in rows)

    gaps = [float(r["score_gap"]) for r in rows]
    diff_counts = [int(r["directive_diff_count"]) for r in rows]
    diff_fracs = [float(r["directive_diff_frac"]) for r in rows]
    lat_gains = [float(r["latency_rel_gain"]) for r in rows if r["obj_mode"] == "PARETO_LATENCY_EXTREME"]
    area_gains = [float(r["area_rel_gain"]) for r in rows if r["obj_mode"] == "PARETO_AREA_EXTREME"]    

    print(f"  total pairs                 : {len(rows)}")
    print(f"  kernel-objective buckets    : {len(by_kernel_obj)}")
    print(f"  pairs by objective          : {dict(by_obj)}")
    print(f"  pairs by tier               : {dict(by_tier)}")

    print(f"  score_gap q10/q50/q90       : {_q(gaps, 0.10):.4f} / {_q(gaps, 0.50):.4f} / {_q(gaps, 0.90):.4f}")
    print(f"  diff_count q10/q50/q90      : {_q(diff_counts, 0.10):.1f} / {_q(diff_counts, 0.50):.1f} / {_q(diff_counts, 0.90):.1f}")
    print(f"  diff_frac q10/q50/q90       : {_q(diff_fracs, 0.10):.3f} / {_q(diff_fracs, 0.50):.3f} / {_q(diff_fracs, 0.90):.3f}")

    if lat_gains:
        print(f"  LATENCY_EXTREME latency gain q10/q50/q90 : {_q(lat_gains, 0.10):.3f} / {_q(lat_gains, 0.50):.3f} / {_q(lat_gains, 0.90):.3f}")
    if area_gains:
        print(f"  AREA_EXTREME area gain q10/q50/q90   : {_q(area_gains, 0.10):.3f} / {_q(area_gains, 0.50):.3f} / {_q(area_gains, 0.90):.3f}")


def preview_preference_pairs(rows: List[dict], n: int = 3) -> None:
    print(f"\n[PAIR-PREVIEW] showing {min(n, len(rows))} pairs")
    for i, ex in enumerate(rows[:n]):
        ch = parse_target_map(ex["chosen"])
        rj = parse_target_map(ex["rejected"])
        changed = [(k, ch.get(k), rj.get(k)) for k in sorted(ch.keys()) if ch.get(k) != rj.get(k)]

        print("\n" + "-" * 100)
        print(f"[{i}] kernel={ex['kernel_name']} obj={ex['obj_mode']}")
        print(f"score_gap={ex['score_gap']:.4f}  diff_count={ex['directive_diff_count']}  diff_frac={ex['directive_diff_frac']:.3f}")
        print(f"latency_rel_gain={ex['latency_rel_gain']:.3f}  area_rel_gain={ex['area_rel_gain']:.3f}")
        print("changed directives:")
        for k, v_ch, v_rj in changed[:12]:
            print(f"  {k}: chosen={v_ch} | rejected={v_rj}")
        if len(changed) > 12:
            print(f"  ... {len(changed) - 12} more")


class DPOPreferenceDataset(Dataset):
    def __init__(
        self,
        mod,
        rows: List[dict],
        tokenizer,
        max_length: int,
        value_weight: float = 1.0,
    ):
        self.mod = mod
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
        self.value_weight = float(value_weight)
        self.samples: List[dict] = []
        self.lengths: List[int] = []

        for ex in rows:
            prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]

            chosen = self._pack_prompt_and_completion(
                prompt_ids=prompt_ids,
                source_text=ex["source_text"],
                completion_text=ex["chosen"],
            )
            rejected = self._pack_prompt_and_completion(
                prompt_ids=prompt_ids,
                source_text=ex["source_text"],
                completion_text=ex["rejected"],
            )

            self.samples.append({
                "kernel_name": ex["kernel_name"],
                "chosen": chosen,
                "rejected": rejected,
            })
            self.lengths.append(max(chosen["length"], rejected["length"]))

    def _pack_prompt_and_completion(self, prompt_ids, source_text: str, completion_text: str) -> Dict[str, torch.Tensor]:
        det_pack = self.mod.build_deterministic_rhs_pack(
            source_text,
            completion_text,
            self.tok,
            value_w=self.value_weight,
        )

        t_ids = det_pack.input_ids
        t_xmask = det_pack.xattn_target_mask

        if len(t_ids) >= self.max_length:
            t_ids = t_ids[: self.max_length]
            t_xmask = t_xmask[: self.max_length]
            prompt_ids_kept = []
        else:
            max_p = self.max_length - len(t_ids)
            prompt_ids_kept = prompt_ids[-max_p:] if len(prompt_ids) > max_p else list(prompt_ids)

        input_ids = prompt_ids_kept + t_ids
        attention_mask = [1] * len(input_ids)

        # DPO compares only RHS tokens (and not prompt / fixed schema).
        score_mask = [0] * len(prompt_ids_kept) + list(t_xmask)

        # Same next-token routing convention as SFT.
        full_xattn_target_mask = [0] * len(prompt_ids_kept) + list(t_xmask)
        xattn_apply_mask = full_xattn_target_mask[1:] + [0]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "score_mask": torch.tensor(score_mask, dtype=torch.float32),
            "xattn_apply_mask": torch.tensor(xattn_apply_mask, dtype=torch.float32),
            "routing_start_idx": torch.tensor(len(prompt_ids_kept), dtype=torch.long),
            "length": len(input_ids),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class DPOPairCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def _pad_1d(self, t: torch.Tensor, max_len: int, pad_value: int | float):
        if t.shape[0] == max_len:
            return t
        pad = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    def __call__(self, batch: List[dict]) -> Dict[str, Any]:
        shared_max_len = 0
        for ex in batch:
            shared_max_len = max(
                shared_max_len,
                ex["chosen"]["input_ids"].shape[0],
                ex["rejected"]["input_ids"].shape[0],
            )

        out = {
            "kernel_name": [ex["kernel_name"] for ex in batch],
        }

        for side in ["chosen", "rejected"]:
            out[f"{side}_input_ids"] = torch.stack([
                self._pad_1d(ex[side]["input_ids"], shared_max_len, self.tok.pad_token_id) for ex in batch
            ])
            out[f"{side}_attention_mask"] = torch.stack([
                self._pad_1d(ex[side]["attention_mask"], shared_max_len, 0) for ex in batch
            ])
            out[f"{side}_score_mask"] = torch.stack([
                self._pad_1d(ex[side]["score_mask"], shared_max_len, 0.0) for ex in batch
            ])
            out[f"{side}_xattn_apply_mask"] = torch.stack([
                self._pad_1d(ex[side]["xattn_apply_mask"], shared_max_len, 0.0) for ex in batch
            ])
            out[f"{side}_routing_start_idx"] = torch.stack([
                ex[side]["routing_start_idx"] for ex in batch
            ])

        return out
    

class HARPDPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        ref_model,
        mem_bank: Dict[str, dict],
        mem_dim: int,
        max_slots: int,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        sft_alpha: float = 0.0,
        group_by_length: bool = False,
        lr_lora: float = 2e-5,
        lr_xattn: float = 5e-5,
        lr_gate: float = 2e-5,
        lr_ff: float = 0.0,
        lr_gate_ff: float = 0.0,
        lr_embed: float = 0.0,
        **kwargs,
    ):
        self.ref_model = ref_model
        self.mem_bank = mem_bank
        self.mem_dim = mem_dim
        self.max_slots = max_slots
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.sft_alpha = float(sft_alpha)
        self._group_by_length = bool(group_by_length)
        self.lr_lora = lr_lora
        self.lr_xattn = lr_xattn
        self.lr_gate = lr_gate
        self.lr_ff = lr_ff
        self.lr_gate_ff = lr_gate_ff
        self.lr_embed = lr_embed
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lora_params, embed_params = [], []
        attn_gate_params, ff_gate_params = [], []
        xattn_attn_params, xattn_ff_params = [], []
        other_trainables = []

        input_emb_param_ids = set()
        output_emb_param_ids = set()

        try:
            emb = self.model.get_input_embeddings()
            input_emb_param_ids = {id(p) for p in emb.parameters()}
        except Exception:
            pass

        try:
            out_emb = self.model.get_output_embeddings()
            if out_emb is not None:
                output_emb_param_ids = {id(p) for p in out_emb.parameters()}
        except Exception:
            pass

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in input_emb_param_ids or id(param) in output_emb_param_ids:
                embed_params.append(param)
            elif "lora_" in name:
                lora_params.append(param)
            elif name.endswith("attn_gate"):
                attn_gate_params.append(param)
            elif name.endswith("ff_gate"):
                ff_gate_params.append(param)
            elif "gated_cross_attn_layer.attn." in name:
                xattn_attn_params.append(param)
            elif "gated_cross_attn_layer.ff." in name:
                xattn_ff_params.append(param)
            else:
                other_trainables.append((name, param))

        opt_groups = []
        if lora_params:
            opt_groups.append({"params": lora_params, "lr": self.lr_lora})
        if embed_params and self.lr_embed > 0:
            opt_groups.append({"params": embed_params, "lr": self.lr_embed})
        if attn_gate_params:
            opt_groups.append({"params": attn_gate_params, "lr": self.lr_gate})
        if ff_gate_params and self.lr_gate_ff > 0:
            opt_groups.append({"params": ff_gate_params, "lr": self.lr_gate_ff})
        if xattn_attn_params:
            opt_groups.append({"params": xattn_attn_params, "lr": self.lr_xattn})
        if xattn_ff_params and self.lr_ff > 0:
            opt_groups.append({"params": xattn_ff_params, "lr": self.lr_ff})
        if other_trainables:
            bad_names = [n for n, _ in other_trainables]
            raise ValueError(
                "[OPT-DPO] Unexpected trainable parameters outside the allowed groups. "
                f"First 20: {bad_names[:20]}"
            )

        try:
            from bitsandbytes.optim import PagedAdamW8bit
            self.optimizer = PagedAdamW8bit(opt_groups, weight_decay=0.0)
        except Exception:
            self.optimizer = torch.optim.AdamW(opt_groups, weight_decay=0.0)

        print(
            f"[OPT-DPO] lora={sum(p.numel() for p in lora_params):,} "
            f"embed={sum(p.numel() for p in embed_params):,} "
            f"attn_gate={sum(p.numel() for p in attn_gate_params):,} "
            f"ff_gate={sum(p.numel() for p in ff_gate_params):,} "
            f"xattn_attn={sum(p.numel() for p in xattn_attn_params):,} "
            f"xattn_ff={sum(p.numel() for p in xattn_ff_params):,}"
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
    
    def _normalize_kname(self, s: str) -> str:
        return re.sub(r"[-\s]+", "_", s.strip().lower())

    def _condition_harp_from_kernel_names(self, model, kernel_names: List[str]):
        kvs, ms = [], []
        for k in kernel_names:
            pack = self.mem_bank.get(k) or self.mem_bank.get(self._normalize_kname(k))
            if pack is None:
                kvs.append(torch.zeros((self.max_slots, self.mem_dim), dtype=torch.float32))
                ms.append(torch.zeros((self.max_slots,), dtype=torch.bool))
            else:
                kvs.append(pack["kv"])
                ms.append(pack["mask"])

        mem_kv = torch.stack(kvs, dim=0)
        mem_m = torch.stack(ms, dim=0)
        device = next(model.parameters()).device
        model.condition_harp(mem_kv.to(device), mem_m.to(device))

    def _clear_harp_if_present(self, model):
        if hasattr(model, "clear_harp"):
            model.clear_harp()

    def _sequence_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        score_mask: torch.Tensor,
        routing_start_idx: torch.Tensor,
        xattn_apply_mask: torch.Tensor,
        kernel_names: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Important:
        - For the TRAINABLE policy model under grad checkpointing, keep HARP conditioned
        until backward finishes.
        - For the frozen ref model / eval no_grad path, clear immediately after forward.
        """
        keep_harp_for_backward = bool(torch.is_grad_enabled() and model.training)

        if hasattr(model, "condition_harp"):
            self._condition_harp_from_kernel_names(model, kernel_names)

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                routing_start_idx=routing_start_idx,
                xattn_apply_mask=xattn_apply_mask,
            )
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            mask = score_mask[:, 1:]

            log_probs = F.log_softmax(logits, dim=-1)
            token_logps = torch.gather(
                log_probs,
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)

            seq_logps = (token_logps * mask).sum(dim=-1)
            token_counts = mask.sum(dim=-1).clamp(min=1.0)
            return seq_logps, token_counts

        finally:
            # DO NOT clear the trainable policy model before backward checkpoint
            # recomputation has happened.
            if not keep_harp_for_backward:
                self._clear_harp_if_present(model)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        try:
            self.accelerator.backward(loss)
        finally:
            # Clear ONLY after backward so checkpoint recomputation still sees HARP memory
            self._clear_harp_if_present(model)
            self._clear_harp_if_present(self.ref_model)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kernel_names = inputs["kernel_name"]

        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_score_mask = inputs["chosen_score_mask"]
        chosen_routing_start_idx = inputs["chosen_routing_start_idx"]
        chosen_xattn_apply_mask = inputs["chosen_xattn_apply_mask"]

        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        rejected_score_mask = inputs["rejected_score_mask"]
        rejected_routing_start_idx = inputs["rejected_routing_start_idx"]
        rejected_xattn_apply_mask = inputs["rejected_xattn_apply_mask"]

        batch_size = chosen_input_ids.shape[0]

        cat_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        cat_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        cat_score_mask = torch.cat([chosen_score_mask, rejected_score_mask], dim=0)
        cat_routing_start_idx = torch.cat([chosen_routing_start_idx, rejected_routing_start_idx], dim=0)
        cat_xattn_apply_mask = torch.cat([chosen_xattn_apply_mask, rejected_xattn_apply_mask], dim=0)
        cat_kernel_names = list(kernel_names) + list(kernel_names)

        pi_logps, pi_token_counts = self._sequence_logps(
            model=model,
            input_ids=cat_input_ids,
            attention_mask=cat_attention_mask,
            score_mask=cat_score_mask,
            routing_start_idx=cat_routing_start_idx,
            xattn_apply_mask=cat_xattn_apply_mask,
            kernel_names=cat_kernel_names,
        )

        with torch.no_grad():
            ref_logps, ref_token_counts = self._sequence_logps(
                model=self.ref_model,
                input_ids=cat_input_ids,
                attention_mask=cat_attention_mask,
                score_mask=cat_score_mask,
                routing_start_idx=cat_routing_start_idx,
                xattn_apply_mask=cat_xattn_apply_mask,
                kernel_names=cat_kernel_names,
            )

        ref_token_counts = ref_token_counts.clamp(min=1.0)
        pi_token_counts = pi_token_counts.clamp(min=1.0)

        pi_scores = pi_logps / pi_token_counts
        ref_scores = ref_logps / ref_token_counts

        pi_chosen, pi_rejected = pi_scores[:batch_size], pi_scores[batch_size:]
        ref_chosen, ref_rejected = ref_scores[:batch_size], ref_scores[batch_size:]
        chosen_token_counts = pi_token_counts[:batch_size]

        preference_logits = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)

        if self.label_smoothing > 0.0:
            losses = (
                -(1.0 - self.label_smoothing) * F.logsigmoid(self.beta * preference_logits)
                - self.label_smoothing * F.logsigmoid(-self.beta * preference_logits)
            )
        else:
            losses = -F.logsigmoid(self.beta * preference_logits)

        loss = losses.mean()

        if self.sft_alpha > 0.0:
            chosen_nll = -pi_chosen
            loss = loss + self.sft_alpha * chosen_nll.mean()

        if return_outputs:
            outputs = {
                "losses": losses.detach(),
                "preference_logits": preference_logits.detach(),
                "pi_chosen": pi_chosen.detach(),
                "pi_rejected": pi_rejected.detach(),
                "ref_chosen": ref_chosen.detach(),
                "ref_rejected": ref_rejected.detach(),
            }
            return loss, outputs
        return loss


    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        is_dpo_batch = (
            isinstance(inputs, dict)
            and "chosen_input_ids" in inputs
            and "rejected_input_ids" in inputs
        )

        if not is_dpo_batch:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)

        try:
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model,
                        inputs,
                        return_outputs=True,
                    )

            loss = loss.mean().detach()

            if prediction_loss_only:
                return loss, None, None

            return loss, None, None

        finally:
            self._clear_harp_if_present(model)
            self._clear_harp_if_present(self.ref_model)


def build_tokenizer(mod, tokenizer_source: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    special_tokens = (
        [g["token"] for g in mod.GOALS.values()]
        + mod.SOURCE_PLACEHOLDER_TOKENS
        + mod.TARGET_PLACEHOLDER_TOKENS
    )
    tok.add_special_tokens({"additional_special_tokens": special_tokens})

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def maybe_restrict_special_token_embeddings(mod, model, tokenizer):
    special_ids = tokenizer.convert_tokens_to_ids(
        [g["token"] for g in mod.GOALS.values()] + mod.SOURCE_PLACEHOLDER_TOKENS + mod.TARGET_PLACEHOLDER_TOKENS
    )

    def enable_only_selected_rows(weight: torch.nn.Parameter, token_ids):
        weight.requires_grad_(True)
        token_ids = torch.tensor(sorted(set(int(x) for x in token_ids if isinstance(x, int) and x >= 0)), dtype=torch.long)

        def grad_mask_hook(grad):
            mask = torch.zeros(grad.size(0), device=grad.device, dtype=grad.dtype)
            mask[token_ids.to(grad.device)] = 1.0
            return grad * mask.unsqueeze(1)

        weight.register_hook(grad_mask_hook)

    inp_emb = model.get_input_embeddings()
    enable_only_selected_rows(inp_emb.weight, special_ids)

    out_emb = model.get_output_embeddings()
    if out_emb is not None and out_emb.weight is not inp_emb.weight:
        enable_only_selected_rows(out_emb.weight, special_ids)



def freeze_embeddings(model):
    try:
        inp = model.get_input_embeddings()
        if inp is not None:
            inp.weight.requires_grad_(False)
    except Exception:
        pass
    try:
        out = model.get_output_embeddings()
        if out is not None:
            out.weight.requires_grad_(False)
    except Exception:
        pass



def configure_dpo_trainables(
    model,
    *,
    train_lora: bool,
    train_xattn: bool,
    train_attn_gate: bool,
    train_ff_gate: bool,
):
    """
    For stage-3 DPO, do NOT let the whole PEFT adapter move by default.
    We want to preserve the stage-2 language prior and let DPO mainly refine
    memory-conditioned routing through HARP xattn.
    """
    model.requires_grad_(False)

    for name, param in model.named_parameters():
        if "lora_" in name and train_lora:
            param.requires_grad_(True)
        elif "gated_cross_attn_layer.attn." in name and train_xattn:
            param.requires_grad_(True)
        elif name.endswith("attn_gate") and train_attn_gate:
            param.requires_grad_(True)
        elif name.endswith("ff_gate") and train_ff_gate:
            param.requires_grad_(True)

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    print(f"[DPO-TRAINABLE] groups enabled: "
          f"lora={train_lora} xattn={train_xattn} attn_gate={train_attn_gate} ff_gate={train_ff_gate}")
    print(f"[DPO-TRAINABLE] total trainable params: {sum(x[1] for x in trainable):,}")
    print(f"[DPO-TRAINABLE] first trainables: {[x[0] for x in trainable[:20]]}")



def build_harp_model(mod, args, tokenizer, trainable: bool):
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
    base.resize_token_embeddings(len(tokenizer))
    if hasattr(base.config, "tie_word_embeddings"):
        base.config.tie_word_embeddings = True
    try:
        base.tie_weights()
    except Exception:
        pass

    base.config.use_cache = False

    gc_enabled = bool(args.gradient_checkpointing and trainable)
    gc_kwargs = {"use_reentrant": False} if gc_enabled else None

    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=gc_enabled,
        gradient_checkpointing_kwargs=gc_kwargs,
    )

    model = PeftModel.from_pretrained(
        base,
        os.path.abspath(args.stage1_adapter_dir),
        is_trainable=trainable,
    )

    mod.extend_instance(model, mod.HARPLMMixin)
    decoder_layers_attr_name = mod.infer_decoder_layers_attr_name(model)
    model.set_decoder_layers_attr_name(decoder_layers_attr_name)

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(mod.TARGET_PLACEHOLDER_TOKENS)
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer LM hidden size from model.config")

    model.init_harp_flamingo(
        placeholder_token_ids=placeholder_token_ids,
        lang_hidden_size=hidden_size,
        mem_hidden_size=args.mem_dim,
        cross_attn_every_n_layers=args.every_n_layers,
        gradient_checkpointing=(args.gradient_checkpointing and trainable),
        xattn_heads=args.xattn_heads,
        xattn_dim_head=args.xattn_dim_head,
        xattn_ff_mult=args.xattn_ff_mult,
        only_attend_immediate_memory=True,
        mask_mode="segment",
    )
    mod.move_harp_modules_to_model_device(model)
    mod.load_partial_harp_xattn(model, args.stage2_harp_xattn_path, tag="HARP-LOAD")

    if trainable:
        configure_dpo_trainables(
            model,
            train_lora=args.train_lora_dpo,
            train_xattn=args.train_xattn_dpo,
            train_attn_gate=args.train_attn_gate_dpo,
            train_ff_gate=args.train_ff_gate_dpo,
        )

        if args.train_special_token_embeddings:
            emb = mod.unfreeze_input_embeddings(model)
            print(f"[TOKENS-DPO] input embeddings unfrozen: {emb.weight.requires_grad}")
            maybe_restrict_special_token_embeddings(mod, model, tokenizer)
        else:
            freeze_embeddings(model)
    else:
        model.requires_grad_(False)
        freeze_embeddings(model)
        model.eval()

    try:
        model.tie_weights()
    except Exception:
        pass

    return model


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--memory_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-7b-base")
    ap.add_argument("--sft_script", type=str, required=True)

    ap.add_argument("--objective", type=str, required=True, choices=[
        "PARETO_LATENCY_EXTREME",
        "PARETO_KNEE",
        "PARETO_AREA_EXTREME",
    ])

    ap.add_argument("--stage1_adapter_dir", type=str, required=True)
    ap.add_argument("--stage2_harp_xattn_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--split_mode", type=str, default="family", choices=["family", "random_design"])
    ap.add_argument("--split_json", type=str, default="")
    ap.add_argument("--save_split_json", type=str, default="")
    ap.add_argument("--val_families", type=str, default="rodinia_pathfinder;machsuite_sort_radix")
    ap.add_argument("--test_families", type=str, default="serrano_kalman_filter")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--stratify_by_kernel", action="store_true")

    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--goal_domination_penalty", type=float, default=0.25)
    ap.add_argument("--goal_max_dominated_gap", type=float, default=0.12)
    ap.add_argument("--score_weight_min", type=float, default=0.6)
    ap.add_argument("--score_weight_power", type=float, default=1.0)

    ap.add_argument("--dpo_chosen_top_k", type=int, default=3)
    ap.add_argument("--dpo_hard_window", type=int, default=8)
    ap.add_argument("--dpo_hard_negatives_per_chosen", type=int, default=2)
    ap.add_argument("--dpo_medium_negatives_per_chosen", type=int, default=1)
    ap.add_argument("--dpo_min_score_gap", type=float, default=0.02)
    ap.add_argument("--dpo_hard_gap_max", type=float, default=0.15)
    ap.add_argument("--dpo_medium_gap_max", type=float, default=0.35)
    ap.add_argument("--dpo_min_primary_rel_gain", type=float, default=0.02)
    ap.add_argument("--dpo_min_edit_distance", type=int, default=1)
    ap.add_argument("--dpo_min_edit_frac", type=float, default=0.0)
    ap.add_argument("--dpo_max_edit_frac", type=float, default=1.0)

    ap.add_argument("--min_supervised_sites", type=int, default=2)
    ap.add_argument("--min_site_coverage", type=float, default=0.85)
    ap.add_argument("--selection_num_val_kernels", type=int, default=6)

    ap.add_argument("--require_same_supervised_schema", dest="require_same_supervised_schema", action="store_true")
    ap.add_argument("--allow_mismatched_supervised_schema", dest="require_same_supervised_schema", action="store_false")
    ap.set_defaults(require_same_supervised_schema=True)

    ap.add_argument("--value_loss_weight", type=float, default=1.0)
    ap.add_argument("--train_special_token_embeddings", action="store_true")
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--sft_alpha", type=float, default=0.0)

    ap.add_argument("--train_lora_dpo", action="store_true")
    ap.add_argument("--train_xattn_dpo", action="store_true")
    ap.add_argument("--train_attn_gate_dpo", action="store_true")
    ap.add_argument("--train_ff_gate_dpo", action="store_true")

    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--group_by_length", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--lr_lora", type=float, default=2e-5)
    ap.add_argument("--lr_xattn", type=float, default=5e-5)
    ap.add_argument("--lr_gate", type=float, default=2e-5)
    ap.add_argument("--lr_ff", type=float, default=0.0)
    ap.add_argument("--lr_gate_ff", type=float, default=0.0)
    ap.add_argument("--lr_embed", type=float, default=0.0)

    ap.add_argument("--mem_dim", type=int, default=32)
    ap.add_argument("--max_slots", type=int, default=64)
    ap.add_argument("--every_n_layers", type=int, default=8)
    ap.add_argument("--xattn_heads", type=int, default=4)
    ap.add_argument("--xattn_dim_head", type=int, default=64)
    ap.add_argument("--xattn_ff_mult", type=int, default=1)

    ap.add_argument("--resume_from_checkpoint", type=str, default="")
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # if the user did not explicitly choose trainable groups,
    # train only HARP xattn + attn_gate.
    if not any([
        args.train_lora_dpo,
        args.train_xattn_dpo,
        args.train_attn_gate_dpo,
        args.train_ff_gate_dpo,
    ]):
        args.train_xattn_dpo = True
        args.train_attn_gate_dpo = True

    mod = import_module_from_path(args.sft_script)

    rows = mod.load_rows(args.dataset)
    print(f"[INFO] Loaded {len(rows)} raw rows from {args.dataset}")
    fam_counts = Counter(r["_family"] for r in rows)
    print("[INFO] Raw rows per family (top 15):", fam_counts.most_common(15))

    train_rows, val_rows, test_rows = build_selected_splits(
        mod=mod,
        args=args,
        rows=rows,
    )

    selected_debug_dir = os.path.join(args.output_dir, "selected_debug")
    os.makedirs(selected_debug_dir, exist_ok=True)
    dump_jsonl(os.path.join(selected_debug_dir, "train_selected.jsonl"), train_rows)
    if val_rows:
        dump_jsonl(os.path.join(selected_debug_dir, "val_selected.jsonl"), val_rows)
    if test_rows:
        dump_jsonl(os.path.join(selected_debug_dir, "test_selected.jsonl"), test_rows)

    print(f"[INFO] Final selected split sizes: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    pair_builder = GoalPreferencePairBuilder(
        mod=mod,
        objective=args.objective,
        chosen_top_k=args.dpo_chosen_top_k,
        hard_window=args.dpo_hard_window,
        hard_negatives_per_chosen=args.dpo_hard_negatives_per_chosen,
        medium_negatives_per_chosen=args.dpo_medium_negatives_per_chosen,
        min_score_gap=args.dpo_min_score_gap,
        hard_gap_max=args.dpo_hard_gap_max,
        medium_gap_max=args.dpo_medium_gap_max,
        min_primary_rel_gain=args.dpo_min_primary_rel_gain,
        min_edit_distance=args.dpo_min_edit_distance,
        min_edit_frac=args.dpo_min_edit_frac,
        max_edit_frac=args.dpo_max_edit_frac,
        min_supervised_sites=args.min_supervised_sites,
        min_site_coverage=args.min_site_coverage,
        require_same_supervised_schema=args.require_same_supervised_schema,
    )
    
    train_pairs = pair_builder.build(train_rows)
    val_pairs = pair_builder.build(val_rows) if val_rows else []
    test_pairs = pair_builder.build(test_rows) if test_rows else []

    if args.objective == "PARETO_KNEE":
        train_pairs = rebalance_knee_pairs(train_pairs, seed=args.seed, max_ratio=1.25)
        val_pairs = rebalance_knee_pairs(val_pairs, seed=args.seed, max_ratio=1.25) if val_pairs else []
        test_pairs = rebalance_knee_pairs(test_pairs, seed=args.seed, max_ratio=1.25) if test_pairs else []

    print(f"[INFO] Preference pairs: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    if len(train_pairs) == 0:
        raise ValueError("No training pairs were constructed. Relax the pair filters.")

    print(f"[INFO] Unique train kernels with pairs: {len(set(r['kernel_name'] for r in train_pairs))}")
    print(f"[INFO] Avg pairs per kernel-objective bucket: {len(train_pairs)/max(1, len(Counter((r['kernel_name'], r['obj_mode']) for r in train_pairs))):.2f}")

    audit_preference_pairs("train", train_pairs)
    audit_preference_pairs("val", val_pairs)
    audit_preference_pairs("test", test_pairs)

    preview_preference_pairs(train_pairs, n=3)

    debug_dir = os.path.join(args.output_dir, "pair_debug")
    os.makedirs(debug_dir, exist_ok=True)
    dump_jsonl(os.path.join(debug_dir, "train_pairs.jsonl"), train_pairs)
    if val_pairs:
        dump_jsonl(os.path.join(debug_dir, "val_pairs.jsonl"), val_pairs)
    if test_pairs:
        dump_jsonl(os.path.join(debug_dir, "test_pairs.jsonl"), test_pairs)

    tokenizer = build_tokenizer(mod, args.model)
    mem_bank = mod.load_memory_bank(args.memory_dir)
    print(f"[INFO] Memory bank keys: {len(mem_bank)}")

    train_ds = DPOPreferenceDataset(
        mod=mod,
        rows=train_pairs,
        tokenizer=tokenizer,
        max_length=args.max_length,
        value_weight=args.value_loss_weight,
    )
    val_ds = DPOPreferenceDataset(
        mod=mod,
        rows=val_pairs,
        tokenizer=tokenizer,
        max_length=args.max_length,
        value_weight=args.value_loss_weight,
    ) if val_pairs else None

    collator = DPOPairCollator(tokenizer)

    print("[INFO] Building policy model...")
    policy_model = build_harp_model(mod, args, tokenizer, trainable=True)
    print("[INFO] Building frozen reference model...")
    ref_model = build_harp_model(mod, args, tokenizer, trainable=False)

    if hasattr(policy_model, "print_trainable_parameters"):
        policy_model.print_trainable_parameters()

    rhs_candidate_bank = mod.build_rhs_candidate_bank(train_rows)

    selection_cases = mod.build_selection_cases(
        val_rows,
        goal_mode=args.objective,
        max_kernels=args.selection_num_val_kernels,
        min_coverage=args.min_site_coverage,
        min_supervised_sites=args.min_supervised_sites,
    )

    effective_total_steps = args.max_steps if args.max_steps > 0 else max(1, math.ceil(len(train_ds) / max(1, args.batch_size * args.grad_accum)) * args.epochs)
    warmup_steps = max(1, int(0.03 * effective_total_steps))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=max(args.lr_lora, args.lr_xattn, args.lr_gate, args.lr_embed, 1e-8),
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = HARPDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        mem_bank=mem_bank,
        mem_dim=args.mem_dim,
        max_slots=args.max_slots,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        sft_alpha=args.sft_alpha,
        group_by_length=args.group_by_length,
        lr_lora=args.lr_lora,
        lr_xattn=args.lr_xattn,
        lr_gate=args.lr_gate,
        lr_ff=args.lr_ff,
        lr_gate_ff=args.lr_gate_ff,
        lr_embed=args.lr_embed,
    )

    trainer.add_callback(mod.SaveHarpXattnCallback())

    if selection_cases:
        trainer.add_callback(
            mod.StageValSelectionCallback(
                tokenizer=tokenizer,
                selection_cases=selection_cases,
                rhs_candidate_bank=rhs_candidate_bank,
                output_dir=args.output_dir,
                max_prompt_tokens=args.max_length,
                candidate_score_reduction="mean",
                best_dir_name="best_custom_stage3",
                mem_bank=mem_bank,
                mem_dim=args.mem_dim,
                max_slots=args.max_slots,
            )
        )

    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        print(f"[INFO] Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    torch.save(
        mod.get_harp_xattn_state_dict(policy_model),
        os.path.join(args.output_dir, "harp_xattn.pt"),
    )

    print(
        f"[DPO-HARP-CONFIG] mem_dim={args.mem_dim} "
        f"max_slots={args.max_slots} "
        f"every_n_layers={args.every_n_layers} "
        f"xattn_heads={args.xattn_heads} "
        f"xattn_dim_head={args.xattn_dim_head} "
        f"xattn_ff_mult={args.xattn_ff_mult}"
    )

    print(f"[DONE] Saved DPO LoRA + HARP xattn adapters to: {args.output_dir}")

    cleanup_cuda()


if __name__ == "__main__":
    main()
