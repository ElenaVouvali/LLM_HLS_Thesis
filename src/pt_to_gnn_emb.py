#-----------------------------------------------------------
#                       Imports 
#-----------------------------------------------------------

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
import torch.nn.functional as F

from config import FLAGS
from model import Net


def load_and_clean_graph(path):
    """Load a single .pt graph and remove any non-tensor attributes."""
    g = torch.load(path, weights_only=False)

    # Inspect the attributes of the Data object
    if isinstance(g, Data):
        keys = list(g.keys())
        # Optionally: drop any non-string keys (usually none in PyG)
        bad_keys = [k for k in keys if not isinstance(k, str)]
        for bk in bad_keys:
            del g[bk]

        # If you had edge_id_to_idx or similar, you can still drop it safely:
        if hasattr(g, "edge_id_to_idx"):
            del g.edge_id_to_idx

    return g



#-----------------------------------------------------------
#               .pt points to graph embeddings
#-----------------------------------------------------------

def extract_single_embedding(pt_point_path, checkpoint_path, device=FLAGS.device):
    """
    pt_point: a ProGraML graph from my dataset
    checkpoint_path: path to trained GNN state_dict (.pth)
    returns: Tensor [d] of frozen embedding (order matches dataset indexing)
    """

    pt_point = load_and_clean_graph(pt_point_path)

    num_features = pt_point.x.size(-1)
    edge_dim = pt_point.edge_attr.size(-1) if getattr(pt_point, "edge_attr", None) is not None else 0

    assert num_features == FLAGS.num_features, f"Feature dim mismatch: {num_features} vs {FLAGS.num_features}"
    assert edge_dim == FLAGS.edge_dim, f"Edge dim mismatch: {edge_dim} vs {FLAGS.edge_dim}"

    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=None).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    # freeze + eval
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    batch = Batch.from_data_list([pt_point]).to(device) # Batch with 1 graph
    with torch.no_grad():
        out = model.forward_embed(batch)   # shape: [batch_size, d]
    
    emb=out.cpu().squeeze(0) 

#    target_indices = ((batch.X_pragmascopenids == 1) | (batch.X_llm_scopeids ==1)).nonzero(as_tuple=True)[0]
#    print(f"Found {len(target_indices)} scope nodes")

#    data = batch
#    P = data.X_pragma_per_node.detach().clone().to(FLAGS.device)
#    P.requires_grad_(True)

    # Plug it back into the data object
#    data.X_pragma_per_node = P

    # Forward (use the real forward that produces perf/area)
#    out_dict, total_loss, loss_dict, gae_loss = model(data)
#    perf_hat = out_dict["perf"].mean()
#    area_hat = out_dict["area"].mean()

    # Scalarize
#    lam = 0.5
#    J = lam * perf_hat + (1 - lam) * area_hat

    # Backprop into P
#    model.zero_grad(set_to_none=True)
#    if P.grad is not None: P.grad.zero_()
#    J.backward()

#    print("dJ/dP shape:", P.grad.shape)      # [N_nodes, 5]

#    for idx in target_indices:
#        node_idx = idx.item()
#        print(f"Node {node_idx} | P: {P[node_idx].detach()} | Grad: {P.grad[node_idx]}")
#        print(f" Node embedding after MLP : {emb[node_idx]}")

    return emb  # shape : [d]

#    print(f"num of layers returned = {len(outs)}")
#    for li, h in enumerate(outs):
        # h: [N, D] where N = num nodes in the (batched) graph
#        h_cpu = h.detach().cpu()
#        print(f"layer {li}: shape={tuple(h_cpu.shape)}")
#        print(f"L9 scope pseudo node : {h_cpu[236, :].tolist()}")
#        nrm = h_cpu[li].norm().item()
#        print(li, "||h|| =", nrm)
#        print("\n")

#    hs = [o[417].detach().cpu() for o in outs]
#    for l in range(1, len(hs)):
#        delta = (hs[l] - hs[l-1]).norm().item()
#        cos = F.cosine_similarity(hs[l].unsqueeze(0), hs[l-1].unsqueeze(0)).item()
#        print(f"{l-1}->{l}: Δ={delta:.4f}, cos={cos:.4f}")

#    with torch.no_grad():
#        outs, attn = model.forward_node_embed_with_attn(batch, capture="all")
#    for rec in attn:
#        ei = rec["edge_index"]   # [2, E']
#        alpha = rec["alpha"]     # [E', heads] or [E']
        # pick edges INTO nid
#        mask = (ei[1] == 417)
#        a = alpha[mask]
#        src = ei[0][mask]
        # average heads if needed
#        if a.dim() == 2:
#            a = a.mean(dim=1)
#        top = torch.topk(a, k=min(10, a.numel()))
#        print(rec["layer"], "top influencers src nodes:", src[top.indices].tolist(), "weights:", top.values.tolist())


#-----------------------------------------------------------
#                   Main Function
#-----------------------------------------------------------

if __name__ == "__main__":

    pt_point_path_1 = "/home/elvouvali/save/harp/pragma-free_kernels/rodinia-knn-1-tiling_processed_result.pt"
#    pt_point_path_1 = "/home/elvouvali/save/harp/rodinia-knn-1-tiling/data_0.pt"
    checkpoint_path = "/home/elvouvali/logs/all_kernels_GNN_train/run1/val_model_state_dict.pth"

    emb = extract_single_embedding(
        pt_point_path_1,
        checkpoint_path,
        device=FLAGS.device
        )


    torch.save(emb, "/home/elvouvali/GNN_embeddings/rodinia-knn-1-tiling-node-embs.pt")
    print(emb.shape)
    print(emb)

#    emb_2 = extract_single_embedding(
#        pt_point_path_2,
#        checkpoint_path,
#        device=FLAGS.device
#        )

#    torch.save(emb_2, "/home/elvouvali/GNN_embeddings/machsuite-gemm-blocked-pred.pt")
#    print(emb_2.shape)
