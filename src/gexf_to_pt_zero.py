"""
gexf_to_pt.py (dataset-compatible)

Goal:
  Produce a .pt that matches the encoding conventions of your original data.py dataset builder.

Key compatibility choices (match data.py):
  - Read GEXF with nx.read_gexf(gexf_path) (no node_type coercion).
  - Enforce numeric-string node ids "0..N-1" (required by _encode_X_dict assertions).
  - Build edge_index using nx.convert_node_labels_to_integers(ordering='sorted') EXACTLY like data.py.
    Note: NetworkX documents this mapping is not guaranteed to be stable across environments. :contentReference[oaicite:2]{index=2}
  - Use the SAME encoders artifact as data.py uses (ENCODER_PATH from data.py).
  - Do NOT call Data.sort() because the original data.py regression branch did not sort edges.

This script does NOT attempt to “fix” any potential misalignment issues in the original dataset pipeline.
It aims to reproduce the dataset representation as-is.
"""

import os
import json
import math
import argparse
import re
import ast
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import networkx as nx

from torch_geometric.data import Data
from collections import defaultdict
from saver import saver
from config import FLAGS

# Import exactly the functions used by the dataset builder
import data as data_py


_ALLOWED_PREFIXES = ("_PIPE_", "_UNROLL_", "_ARRAY_T_", "_ARRAY_F_", "_ARRAY_D_")
_ARRAY_TMAP = {"cyclic": 100, "block": 200, "complete": 300}
_AUTO_RE = re.compile(r'auto\{([^}]+)\}')
_LABEL_RE = re.compile(r'_L(\d+)\b')


def _zero_point_from_gexf(g: nx.Graph) -> Dict[str, int]:
    """
    Extract every auto{_PIPE_...}, auto{_UNROLL_...}, auto{_ARRAY_*_...} from node full_text
    and assign 0. This makes _encode_X_dict safe even without a real point-json.
    """
    point = {}
    for _, ndata in g.nodes(data=True):
        ft = ndata.get("full_text", "")
        if not isinstance(ft, str):
            continue
        for key in _AUTO_RE.findall(ft):
            if key.startswith(_ALLOWED_PREFIXES):
                point[key] = 0
    return point


def _baseline_point_from_gexf(g: nx.Graph) -> Dict[str, Any]:
    """
    Build an in-distribution constant baseline P0 for every auto{...} placeholder found in full_text.
    """
    point = {}
    for _, ndata in g.nodes(data=True):
        ft = ndata.get("full_text", "")
        if not isinstance(ft, str):
            continue
        for key in _AUTO_RE.findall(ft):
            if not key.startswith(_ALLOWED_PREFIXES):
                continue

            if key.startswith("_PIPE_"):
                point[key] = 1                      # II=1
            elif key.startswith("_UNROLL_"):
                point[key] = 1                      # unroll=1
            elif key.startswith("_ARRAY_D_"):
                point[key] = 1                      # dim=1
            elif key.startswith("_ARRAY_F_"):
                point[key] = 1                      # factor=1
            elif key.startswith("_ARRAY_T_"):
                point[key] = "block"                # will map to 200 in your vector builder
            else:
                point[key] = 0
    return point



def _require_numeric_string_nodes(g: nx.Graph) -> None:
    """
    data.py's _encode_X_dict() sorts nodes by int(node_id) and asserts:
      nid == int(node)
    Therefore, node ids must be numeric strings forming 0..N-1 with no gaps.
    """
    nodes = list(g.nodes())
    for n in nodes:
        if not isinstance(n, str) or not n.isdigit():
            raise RuntimeError(
                f"Dataset-compatible encoding requires numeric-string node ids. "
                f"Found node id: {n!r} (type={type(n)})"
            )

    ints = sorted(int(n) for n in nodes)
    if not ints:
        raise RuntimeError("Graph has no nodes.")
    if ints[0] != 0 or ints[-1] != len(ints) - 1:
        raise RuntimeError(
            f"Node ids must be contiguous 0..N-1. "
            f"Got min={ints[0]}, max={ints[-1]}, N={len(ints)}."
        )


def create_edge_index_dataset_compatible(g: nx.Graph) -> torch.Tensor:
    """
    EXACT copy of data.py's create_edge_index() behavior:
      g2 = nx.convert_node_labels_to_integers(g, ordering='sorted')
      edge_index = torch.LongTensor(list(g2.edges)).t().contiguous()

    Important: This can yield different mappings across environments per NetworkX docs. :contentReference[oaicite:3]{index=3}
    """
    g2 = nx.convert_node_labels_to_integers(g, ordering="sorted")
    return torch.LongTensor(list(g2.edges)).t().contiguous()


def _normalize_point_for_pragmas_vector(point: Dict[str, Any]) -> Dict[str, Any]:
    """
    For compatibility:
      - Keep point mostly as provided for _encode_X_dict() (it can handle strings in many places).
      - BUT when building the flat 'pragmas' vector, data.py converts array type tokens.
    This helper just ensures JSON oddities (None, floats-as-ints) behave.
    """
    out: Dict[str, Any] = {}
    for k, v in point.items():
        if v is None:
            out[k] = 0
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, float) and float(v).is_integer():
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _build_pragmas_vector(point: Dict[str, Any], max_pragma_length: int) -> torch.Tensor:
    """
    Match data.py behavior used for the stored 'pragmas' tensor:
      - include only keys containing _PIPE_ / _UNROLL_ / _ARRAY_
      - iterate in sorted(point.items()) order
      - convert _ARRAY_T_ values from string -> {100,200,300}
      - convert other string values to int
      - pad to max_pragma_length with zeros
    """
    pragmas = []
    for name, value in sorted(point.items()):
        if not name.startswith(_ALLOWED_PREFIXES):
            continue

        if isinstance(value, str):
            v = value.strip().lower()
            if name.startswith("_ARRAY_T_"):
                value = _ARRAY_TMAP.get(v, 0)
            else:
                try:
                    value = int(v)
                except ValueError:
                    raise ValueError(f"Non-numeric pragma value '{value}' for key {name}")
        elif not isinstance(value, int):
            raise ValueError(f"Unexpected pragma value type: {type(value)} for key {name}")

        pragmas.append(value)

    if len(pragmas) > max_pragma_length:
        raise RuntimeError(
            f"Pragmas length {len(pragmas)} exceeds max_pragma_length {max_pragma_length}. "
            "To match dataset, load the correct pragma_dim file or use the dataset's max."
        )

    pragmas.extend([0] * (max_pragma_length - len(pragmas)))
    return torch.FloatTensor(np.array([pragmas], dtype=np.float32))


def _fill_targets_like_data_py(xy_dict: Dict[str, Any], perf_val: float, area_val: float) -> None:
    """
    Create regression targets in the same normalization style as data.py.
    For LLM predictions you may not have real perf/area; defaults are acceptable.
    """
    if getattr(FLAGS, "task", "regression") != "regression":
        xy_dict["perf"] = torch.LongTensor([0])
        return

    eps = float(getattr(FLAGS, "epsilon", 1e-6))
    norm_method = getattr(FLAGS, "norm_method", "log2")

    def norm(v: float) -> float:
        if norm_method == "log2":
            return math.log2(v + eps)
        if norm_method == "const":
            return v * float(getattr(FLAGS, "normalizer", 1.0))
        if norm_method == "off":
            return v
        if "speedup" in norm_method:
            # In data.py speedup uses FLAGS.normalizer / perf
            if v <= 0:
                return 0.0
            speedup = float(getattr(FLAGS, "normalizer", 1.0)) / v
            if norm_method == "speedup-log2":
                return math.log2(speedup + eps)
            return speedup
        # fallback
        return math.log2(v + eps)

    # perf
    xy_dict["perf"] = torch.FloatTensor(np.array([norm(perf_val)], dtype=np.float32))
    xy_dict["actual_perf"] = torch.FloatTensor(np.array([perf_val], dtype=np.float32))
    xy_dict["kernel_speedup"] = torch.FloatTensor(np.array([0.0], dtype=np.float32))

    # area
    area_safe = area_val if area_val > 0.0 else eps
    if norm_method == "const":
        area_norm = area_safe * float(getattr(FLAGS, "util_normalizer", 1.0))
    elif norm_method == "off":
        area_norm = area_safe
    else:
        area_norm = math.log2(area_safe + eps)

    xy_dict["area"] = torch.FloatTensor(np.array([area_norm], dtype=np.float32))
    xy_dict["actual_area"] = torch.FloatTensor(np.array([area_safe], dtype=np.float32))



def _as_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_label_id_from_full_text(ft: Any) -> int:
    """
    Extract label id from strings like:
      "... auto{_PIPE_L3} ..." or "... auto{_ARRAY_T_L7} ..."
    Returns integer (e.g., 3, 7) or -1 if not found.
    """
    if not isinstance(ft, str):
        return -1
    # try direct pattern first
    m = _LABEL_RE.search(ft)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    for key in _AUTO_RE.findall(ft):
        mm = _LABEL_RE.search(key)
        if mm:
            try:
                return int(mm.group(1))
            except Exception:
                return -1
    return -1



def _build_llm_scope_tensors(g: nx.Graph) -> Dict[str, torch.Tensor]:
    """
    Build per-node masks for LLM memory anchors.

      - Loop pragmas (PIPELINE / UNROLL): anchor on the pseudo-block scope node
      - Array pragmas (ARRAY_PARTITION): anchor on the pragma node itself

    """
    N = g.number_of_nodes()
    scopeids = torch.zeros((N,), dtype=torch.long)
    scopecat = torch.zeros((N,), dtype=torch.long)   # 0 none, 1 loop-scope pseudo, 2 array-pragma node
    label_ids = torch.full((N,), -1, dtype=torch.long)

    def _extract_label_ids_from_full_text(ft: Any) -> List[int]:

        if not isinstance(ft, str):
            return []

        lids = set()

        # scan actual auto{...} placeholders and keep only allowed prefixes
        for key in _AUTO_RE.findall(ft):
            if key.startswith(_ALLOWED_PREFIXES):
                m = _LABEL_RE.search(key)
                if m:
                    lids.add(int(m.group(1)))

        # Fallback: if nothing found, scan whole text for Lk labels
        if not lids:
            for m in _LABEL_RE.finditer(ft):
                lids.add(int(m.group(1)))

        return sorted(lids)

    # Deterministic map: (block, function) -> pseudo node id
    # Keep the smallest nid if multiple pseudo nodes share the same (block, function)
    pseudo_by_bf = {}
    for nid_str, ndata in g.nodes(data=True):
        if not (isinstance(nid_str, str) and nid_str.isdigit()):
            continue

        nid = int(nid_str)
        ntype = _as_int(ndata.get("type", -1), -1)
        text = str(ndata.get("text", "")).lower()

        if ntype == 4 or "pseudo" in text:
            b = _as_int(ndata.get("block", -1), -1)
            f = _as_int(ndata.get("function", -1), -1)
            if b != -1 and f != -1:
                prev = pseudo_by_bf.get((b, f))
                if prev is None or nid < prev:
                    pseudo_by_bf[(b, f)] = nid

    # Collect placeholder -> anchor assignments first
    # This avoids silent overwrites and keeps behavior deterministic.
    assignments = []   # list of (label_id, anchor_nid, anchor_cat)

    for nid_str, ndata in sorted(g.nodes(data=True), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 10**18):
        if not (isinstance(nid_str, str) and nid_str.isdigit()):
            continue

        nid = int(nid_str)
        ntype = _as_int(ndata.get("type", -1), -1)
        if ntype != 100:
            continue

        kind = str(ndata.get("text", "")).strip().upper()
        ft = ndata.get("full_text", "")
        lids = _extract_label_ids_from_full_text(ft)

        if not lids:
            continue

        b = _as_int(ndata.get("block", -1), -1)
        f = _as_int(ndata.get("function", -1), -1)

        if kind in ("PIPELINE", "UNROLL"):
            pseudo = pseudo_by_bf.get((b, f), None)
            if pseudo is None:
                raise RuntimeError(
                    f"Could not find pseudo scope node for loop pragma node {nid} "
                    f"(kind={kind}, block={b}, function={f}, full_text={ft!r})"
                )
            for lid in lids:
                assignments.append((lid, pseudo, 1))

        elif kind == "ARRAY_PARTITION":
            for lid in lids:
                assignments.append((lid, nid, 2))

    # Enforce one unique anchor per label and one unique label per anchor
    label_to_anchor = {}
    anchor_to_label = {}

    for lid, anchor_nid, anchor_cat in assignments:
        # Same label appearing multiple times with different anchors is inconsistent
        if lid in label_to_anchor:
            prev_nid, prev_cat = label_to_anchor[lid]
            if prev_nid != anchor_nid or prev_cat != anchor_cat:
                raise RuntimeError(
                    f"Placeholder L{lid} maps to multiple anchors: "
                    f"({prev_nid}, cat={prev_cat}) and ({anchor_nid}, cat={anchor_cat})."
                )
        else:
            label_to_anchor[lid] = (anchor_nid, anchor_cat)

        # Different labels mapping to the same anchor cannot be represented
        # with the current per-node X_llm_labelid format
        if anchor_nid in anchor_to_label:
            prev_lid = anchor_to_label[anchor_nid]
            if prev_lid != lid:
                raise RuntimeError(
                    f"Anchor node {anchor_nid} corresponds to multiple placeholders "
                    f"(L{prev_lid} and L{lid}). With the current per-node tensors "
                    f"you cannot get one memory token per placeholder in this case."
                )
        else:
            anchor_to_label[anchor_nid] = lid

    # Fill output tensors
    for lid, (anchor_nid, anchor_cat) in label_to_anchor.items():
        scopeids[anchor_nid] = 1
        scopecat[anchor_nid] = anchor_cat
        label_ids[anchor_nid] = lid

    return {
        "X_llm_scopeids": scopeids,
        "X_llm_scopecat": scopecat,
        "X_llm_labelid": label_ids,
    }



def gexf_to_pt(gexf_path: str, point_json: str, out_pt: str, key_name: str,
              perf: float = 0.0, area: float = 0.0, max_pragma_length: int = 93) -> None:
    # Load encoders exactly as data.py does
    encoders = data_py.load_encoders()
    enc_ntype = encoders["enc_ntype"]
    enc_ptype = encoders["enc_ptype"]
    enc_itype = encoders["enc_itype"]
    enc_ftype = encoders["enc_ftype"]
    enc_btype = encoders["enc_btype"]
    enc_ftype_edge = encoders["enc_ftype_edge"]
    enc_ptype_edge = encoders["enc_ptype_edge"]

    # Read graph the same way dataset builder does
    gexf_path = os.path.expanduser(gexf_path)
    g = nx.read_gexf(gexf_path)

    _require_numeric_string_nodes(g)

    gname = os.path.basename(gexf_path).split(".")[0]
    new_gname = gname.split("_")[0]

    # Load point JSON (optional). If missing/None/"NONE", build zeros point from GEXF.
    if point_json is None or str(point_json).upper() == "NONE":
        point = _zero_point_from_gexf(g)
#    if point_json is None or str(point_json).upper() == "NONE":
#        point = _baseline_point_from_gexf(g)
    else:
        with open(os.path.expanduser(point_json), "r", encoding="utf-8") as f:
            point_raw = json.load(f)
        point = _normalize_point_for_pragmas_vector(point_raw)

    # Build edge_index exactly like dataset
    edge_index = create_edge_index_dataset_compatible(g)

    # These follow dataset behavior (even if edge_index is built from a relabeled copy)
    edge_id_to_idx = data_py.build_edge_id_to_idx(g)
    edge_dict = data_py._encode_edge_dict(g, ftypes=None, ptypes=None)

    # Encode node features exactly like dataset
    xy_dict = data_py._encode_X_dict(
        g,
        ntypes=None,
        ptypes=None,
        numerics=None,
        itypes=None,
        ftypes=None,
        btypes=None,
        point=point,
    )

    # Force "pragma-free values" representation:
    xy_dict["X_pragma_per_node"] = torch.zeros_like(xy_dict["X_pragma_per_node"])
    xy_dict["pragmas"] = torch.zeros((1, max_pragma_length), dtype=torch.float32)

    # Build LLM scope masks from the GEXF (does NOT affect GNN compatibility)
    llm_scope = _build_llm_scope_tensors(g)

    # Pragmas vector stored as attribute
#   xy_dict["pragmas"] = _build_pragmas_vector(point, max_pragma_length=max_pragma_length)

    # Targets (defaults are okay if you only care about graph tensors)
    _fill_targets_like_data_py(xy_dict, perf_val=float(perf), area_val=float(area))

    # Dense matrices
    X = data_py._encode_X_torch(xy_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
    edge_attr = data_py._encode_edge_torch(edge_dict, enc_ftype_edge, enc_ptype_edge)

    # Build Data object with same field names used by dataset
    data_obj = Data(
        gname=new_gname,
        x=X,
        key=key_name,
        edge_index=edge_index,
        edge_attr=edge_attr,
        kernel=gname,
        X_contextnids=xy_dict["X_contextnids"],
        X_pragmanids=xy_dict["X_pragmanids"],
        X_pragmascopenids=xy_dict["X_pragmascopenids"],
        X_pseudonids=xy_dict["X_pseudonids"],
        X_icmpnids=xy_dict["X_icmpnids"],
        X_pragma_per_node=xy_dict["X_pragma_per_node"],
        pragmas=xy_dict["pragmas"],
        perf=xy_dict["perf"],
        edge_id_to_idx=edge_id_to_idx,
        X_llm_scopeids=llm_scope["X_llm_scopeids"],
        X_llm_scopecat=llm_scope["X_llm_scopecat"],
        X_llm_labelid=llm_scope["X_llm_labelid"],
    )

    # Include commonly present fields (safe even if you compare strictly)
    if "actual_perf" in xy_dict:
        data_obj.actual_perf = xy_dict["actual_perf"]
    if "kernel_speedup" in xy_dict:
        data_obj.kernel_speedup = xy_dict["kernel_speedup"]
    if "area" in xy_dict:
        data_obj.area = xy_dict["area"]
    if "actual_area" in xy_dict:
        data_obj.actual_area = xy_dict["actual_area"]

    # IMPORTANT: do not call data_obj.sort() to match dataset build (regression path)
    out_pt = os.path.expanduser(out_pt)
    os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
    torch.save(data_obj, out_pt)

    print(f"[OK] Saved: {out_pt}")
    #print("Shapes:")
    #print("  x         :", tuple(data_obj.x.shape))
    #print("  edge_index:", tuple(data_obj.edge_index.shape))
    #print("  edge_attr :", tuple(data_obj.edge_attr.shape))
    #print("  nonzero(pragmas):", int(torch.count_nonzero(data_obj.pragmas)))
    #print("  nonzero(X_pragma_per_node):", int(torch.count_nonzero(data_obj.X_pragma_per_node)))
    #print("  sum(|pragmas|):", float(data_obj.pragmas.abs().sum()))
    #print("  sum(|X_pragma_per_node|):", float(data_obj.X_pragma_per_node.abs().sum()))


#gexf_to_pt(
#    gexf_path='/home/elvouvali/LLM_predictions/rodinia-knn-2-pipeline_connected_hierarchy.gexf',
#    point_json='/home/elvouvali/LLM_predictions/llm_pred_knn_2_pipeline_289.json',
#    out_pt='/home/elvouvali/LLM_predictions/rodinia-knn-2-pipeline-pred.pt',
#    key_name='LLM_pred',
#    )



