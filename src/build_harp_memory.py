import argparse
import glob
import os
import re
import torch

from os.path import join, basename
from torch_geometric.data import Batch

from config import FLAGS
from model import Net
from pt_to_gnn_emb import load_and_clean_graph


def _normalize_kernel_name(s: str) -> str:
    return re.sub(r"[-\s]+", "_", s.strip().lower())    # match both '-' and '_'



@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_path", default="/home/elvouvali/save/harp/pragma-free_kernels")
    ap.add_argument("--ckpt", default="/home/elvouvali/logs/all_kernels_GNN_train/run1/val_model_state_dict.pth")
    ap.add_argument("--out", default="/home/elvouvali/save/harp/memory_tokens")

    # Leakage control
    ap.add_argument("--disable_pragma_injection", action="store_true",
                    help="Recommended: set X_pragmascopenids=0 before running the GNN (pure-structure embeddings).")

    ap.add_argument("--max_slots", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pt_files = glob.glob(join(args.pt_path, "*_processed_result.pt"))
    if not pt_files:
        print(f"No files found in {args.pt_path}")
        return
    
    first_pt = load_and_clean_graph(pt_files[0])
    num_features = first_pt.x.size(-1)
    edge_dim = first_pt.edge_attr.size(-1) if getattr(first_pt, "edge_attr", None) is not None else 0

    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=None).to(FLAGS.device)
    state = torch.load(args.ckpt, map_location=FLAGS.device)
    model.load_state_dict(state)
    model.eval()

    print(f"Starting processing of {len(pt_files)} files...")

    targets = ["rodinia", "machsuite", "spcl", "serrano"]
    
    for kernel_path in pt_files:
        # {x}_processed_result.pt -> {x}.memory.pt
        fname = basename(kernel_path)
        normalized_name = _normalize_kernel_name(fname)
        if not any(t in normalized_name for t in targets):
            continue
        prefix_match = re.search(r"(.+)_processed_result\.pt", fname)
        if prefix_match:
            base_name = prefix_match.group(1)

        output_path = join(args.out, f"{base_name}.memory.pt")

        try:
            pt_point = load_and_clean_graph(kernel_path)
            batch = Batch.from_data_list([pt_point]).to(FLAGS.device)

            graph_embed = model.forward_embed(batch)

            # if hasattr(batch, "X_pragmascopenids"):
            #     scope_mask_orig = (batch.X_pragmascopenids > 0.5).detach().clone()
            # else:
            #     print(f"Skipping {fname}: Missing X_pragmascopenids")
            #     continue

            # disable pragma injection by zeroing the scope mask used by apply_pragma_mlp() 
            # => prevent any pragma-conditioned updates inside the GNN
            if args.disable_pragma_injection:
                batch.X_pragmascopenids = torch.zeros_like(batch.X_pragmascopenids)

            node_emb = model.forward_node_embed(batch)

#            # Use X_llm_scopeids and X_llm_labelid to build slot-aligned memory
            scope = batch.X_llm_scopeids.bool()
            label = batch.X_llm_labelid.long()

            # tokens that correspond to placeholders
            sel = scope & (label > 0) & (label <= args.max_slots)
            sel_idx = sel.nonzero(as_tuple=False).view(-1)

            # slot-aligned outputs
            node_embs = torch.zeros((args.max_slots, node_emb.size(-1)), dtype=node_emb.dtype, device=node_emb.device)
            node_embs_mask = torch.zeros((args.max_slots,), dtype=torch.bool, device=node_emb.device)

            node_ids = [-1] * args.max_slots
            labels = [-1] * args.max_slots

            for ni in sel_idx.tolist():
                lid = int(label[ni].item())  # 1..max_slots
                slot = lid - 1
                node_embs[slot] = node_emb[ni]
                node_embs_mask[slot] = True
                node_ids[slot] = ni
                labels[slot] = lid

            node_embs = node_embs.detach().cpu()
            node_embs_mask = node_embs_mask.detach().cpu()
            graph_embed = graph_embed.detach().cpu()

            node_embs = torch.nan_to_num(node_embs, nan=0.0, posinf=0.0, neginf=0.0)
            max_norm = 20.0
            eps = 1e-6
            norms = node_embs.norm(p=2, dim=1, keepdim=True).clamp(min=eps) # L2 normalization
            scale = (max_norm / norms).clamp(max=1.0)
            node_embs = node_embs * scale

            pack = {
                "pt_path": kernel_path,
                "ckpt": args.ckpt,
                "disable_pragma_injection": bool(args.disable_pragma_injection),
                "gnn_dim": int(node_embs.size(-1)),
                "node_embs": node_embs,
                "node_embs_mask": node_embs_mask,
                "graph_embed": graph_embed,
                "max_slots": args.max_slots,
                "slot_ids": torch.arange(1, args.max_slots + 1, dtype=torch.long),
                "node_ids": node_ids,
                "labels": labels
                }

            torch.save(pack, output_path)
            print(f"[OK] {fname} -> Node embeddings shape : {node_embs.shape}")
            
            # Select only rows where memory_mask is True
            active_memory = node_embs[node_embs_mask] 
            
            if active_memory.size(0) > 0:
                print(f"node_embs[node_embs_mask] --> shape: {active_memory.shape}")
                print(active_memory)
 #               print(f"Global graph embedding : {graph_embed}")
                print(f"node_embs_mask : {node_embs_mask}")
                print(f"node_ids : {node_ids}")
                print(f"labels : {labels}")
            else:
                print("Kernel found, but no slots were identified (All zeros).")


        except Exception as e:
            print(f"[ERROR] Failed to process {fname}: {e}")



if __name__ == "__main__":
    main()

