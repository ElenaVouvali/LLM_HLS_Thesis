import torch
import os
import glob
from gexf_to_pt_zero import gexf_to_pt

GEXF_DIR = "/home/ubuntu/harp/processed/extended-pseudo-block-connected-hierarchy"
OUT_DIR  = "/home/ubuntu/save/harp/pragma-free_kernels"

os.makedirs(OUT_DIR, exist_ok=True)

for gexf_path in sorted(glob.glob(os.path.join(GEXF_DIR, "*.gexf"))):
    base = os.path.basename(gexf_path).replace(".gexf", "")
    out_pt = os.path.join(OUT_DIR, f"{base}.pt")
    gexf_to_pt(
        gexf_path=gexf_path,
        point_json="NONE",          # triggers auto zero-point from GEXF
        out_pt=out_pt,
        key_name="pragma_free",
        perf=0.0,
        area=0.0,
        max_pragma_length=93
    )
    print("Saved", out_pt)


bad = []
for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.pt"))):
    d = torch.load(p, map_location="cpu", weights_only=False)

    nz_pragmas = int(torch.count_nonzero(d.pragmas)) if hasattr(d, "pragmas") else None
    nz_node    = int(torch.count_nonzero(d.X_pragma_per_node)) if hasattr(d, "X_pragma_per_node") else None

    if nz_pragmas != 0 or nz_node != 0:
        bad.append((p, f"nonzero(pragmas)={nz_pragmas}, nonzero(X_pragma_per_node)={nz_node}"))
        continue

    required = [
        "X_arrayscopenids",
        "X_pipeline_scopeids",
        "X_unroll_scopeids",
        "X_array_partition_scopeids",
        "X_scopenids",
        "X_llm_scopeids",
        "X_llm_scopecat",
        "X_llm_labelid",
    ]
    missing = [k for k in required if not hasattr(d, k)]
    if missing:
        bad.append((p, f"missing fields: {missing}"))
        continue

    llm_scope = d.X_llm_scopeids.bool()
    pseudo = d.X_pseudonids.bool()
    array_scope = d.X_arrayscopenids.bool()
    scope_cat = d.X_llm_scopecat.long()

    bad_loop_anchor = llm_scope & (scope_cat == 1) & (~pseudo)
    bad_array_anchor = llm_scope & (scope_cat == 2) & (~array_scope)

    if bad_loop_anchor.any() or bad_array_anchor.any():
        bad.append((p, "LLM scope anchors do not match pseudo/array_scope nodes"))
        continue

    if bad:
        print("FAILED:")
        for p, msg in bad:
            print(" ", p, "->", msg)
        raise SystemExit(1)


print("PASS: all .pt files have pragmas==0 and X_pragma_per_node==0")

