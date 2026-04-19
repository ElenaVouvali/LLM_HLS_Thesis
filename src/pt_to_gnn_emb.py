#-----------------------------------------------------------
#                       Imports 
#-----------------------------------------------------------

import torch
from torch_geometric.data import Batch, Data

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


def load_frozen_gnn(checkpoint_path, sample_graph, device=FLAGS.device):
    num_features = sample_graph.x.size(-1)
    edge_dim = sample_graph.edge_attr.size(-1) if getattr(sample_graph, "edge_attr", None) is not None else 0

    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=None).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def disable_pragma_conditioning(data):
    for name in (
        "X_pragmascopenids",
        "X_pipeline_scopeids",
        "X_unroll_scopeids",
        "X_array_partition_scopeids",
    ):
        if hasattr(data, name):
            setattr(data, name, torch.zeros_like(getattr(data, name)))

    if hasattr(data, "X_pragma_per_node"):
        data.X_pragma_per_node = torch.zeros_like(data.X_pragma_per_node)
    if hasattr(data, "pragmas"):
        data.pragmas = torch.zeros_like(data.pragmas)

    return data


#-----------------------------------------------------------
#               .pt points to graph embeddings
#-----------------------------------------------------------

@torch.no_grad()
def extract_single_embedding(
    pt_point_path,
    checkpoint_path,
    device=FLAGS.device,
    disable_pragma_injection=True,
):
    pt_point = load_and_clean_graph(pt_point_path)
    model = load_frozen_gnn(checkpoint_path, pt_point, device=device)

    batch = Batch.from_data_list([pt_point]).to(device)
    if disable_pragma_injection:
        batch = disable_pragma_conditioning(batch)

    emb = model.forward_embed(batch).detach().cpu().squeeze(0)
    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    return emb


@torch.no_grad()
def extract_slot_aligned_memory(
    pt_point_path,
    checkpoint_path,
    max_slots=64,
    device=FLAGS.device,
    disable_pragma_injection=True,
):
    pt_point = load_and_clean_graph(pt_point_path)
    model = load_frozen_gnn(checkpoint_path, pt_point, device=device)

    batch = Batch.from_data_list([pt_point]).to(device)
    if disable_pragma_injection:
        batch = disable_pragma_conditioning(batch)

    graph_embed = model.forward_embed(batch)
    node_emb = model.forward_node_embed(batch)

    scope = batch.X_llm_scopeids.bool()
    label = batch.X_llm_labelid.long()
    scopecat = batch.X_llm_scopecat.long()

    sel = scope & (label > 0) & (label <= max_slots)
    sel_idx = sel.nonzero(as_tuple=False).view(-1)

    node_embs = torch.zeros((max_slots, node_emb.size(-1)), dtype=node_emb.dtype, device=node_emb.device)
    node_embs_mask = torch.zeros((max_slots,), dtype=torch.bool, device=node_emb.device)
    slot_cats = torch.zeros((max_slots,), dtype=torch.long, device=node_emb.device)

    node_ids = [-1] * max_slots
    labels = [-1] * max_slots

    for ni in sel_idx.tolist():
        lid = int(label[ni].item())
        slot = lid - 1
        node_embs[slot] = node_emb[ni]
        node_embs_mask[slot] = True
        slot_cats[slot] = int(scopecat[ni].item())
        node_ids[slot] = ni
        labels[slot] = lid

    node_embs = torch.nan_to_num(node_embs.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    graph_embed = torch.nan_to_num(graph_embed.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    node_embs_mask = node_embs_mask.detach().cpu()
    slot_cats = slot_cats.detach().cpu()

    return {
        "node_embs": node_embs,
        "node_embs_mask": node_embs_mask,
        "graph_embed": graph_embed,
        "slot_cats": slot_cats,
        "node_ids": node_ids,
        "labels": labels,
    }


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
