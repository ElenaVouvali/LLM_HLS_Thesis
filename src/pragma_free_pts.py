import os
import glob
from gexf_to_pt_zero import gexf_to_pt

GEXF_DIR = "/home/zervakis/elvouvali/harp/processed/extended-pseudo-block-connected-hierarchy"
OUT_DIR  = "/home/zervakis/elvouvali/save/harp/pragma-free_kernels"

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


import torch

bad = []
for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.pt"))):
    d = torch.load(p, map_location="cpu", weights_only=False)
    nz_pragmas = int(torch.count_nonzero(d.pragmas)) if hasattr(d, "pragmas") else None
    nz_node    = int(torch.count_nonzero(d.X_pragma_per_node)) if hasattr(d, "X_pragma_per_node") else None

    if nz_pragmas != 0 or nz_node != 0:
        bad.append((p, nz_pragmas, nz_node))

print("Checked:", len(glob.glob(os.path.join(OUT_DIR, '*.pt'))))
if bad:
    print("FAILED (not pragma-free):")
    for p, a, b in bad:
        print(" ", p, "nonzero(pragmas)=", a, "nonzero(X_pragma_per_node)=", b)
    raise SystemExit(1)

print("PASS: all .pt files have pragmas==0 and X_pragma_per_node==0")

