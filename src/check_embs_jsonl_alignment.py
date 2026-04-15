
import json
import torch
from data import MyOwnDataset

EMB_PATH   = "/home/elvouvali/GNN_embeddings/all_kernels_gnn_embeddings.pt"
JSONL_PATH = "/home/elvouvali/LLM_data/all_kernels_llm_data_filtered.jsonl"
GOOD_FILES = "good_files.txt"

def main():
    # Load embeddings
    embs = torch.load(EMB_PATH, map_location="cpu")
    print("Embeddings shape:", embs.shape)

    # Rebuild the SAME dataset as in extract_GNN_embs.py
    try:
        with open(GOOD_FILES) as f:
            good_files = [line.strip() for line in f if line.strip()]
        dataset = MyOwnDataset(data_files=good_files)
        print(f"Dataset size: {len(dataset)} graphs (filtered)")
    except FileNotFoundError:
        dataset = MyOwnDataset()
        print(f"Dataset size: {len(dataset)} graphs (all)")

    # Count JSONL lines
    with open(JSONL_PATH, "r") as f:
        json_lines = [line for line in f if line.strip()]
    print("JSONL lines:", len(json_lines))

    # Check the lengths
    assert embs.shape[0] == len(dataset) == len(json_lines), \
        "Mismatch between embeddings, dataset, and JSONL sizes!"

    print("\nSizes are consistent")

    # Check a few random indices for kernel + csvrow consistency
    import random
    indices_to_check = [0, 10, 123, 500, 2000, len(dataset)-1]
    indices_to_check = [i for i in indices_to_check if i < len(dataset)]

    print("\nSpot-checking a few indices:")
    for i in indices_to_check:
        g = dataset[i]
        json_obj = json.loads(json_lines[i])

        # Reconstruct the key used during filtering:
        kernel_name = g.kernel
        suffix = "_processed_result"
        if kernel_name.endswith(suffix):
            kernel_name = kernel_name[:-len(suffix)]

        raw = g.key if isinstance(g.key, str) else str(g.key)
        csvrow_idx = int(raw.split("csvrow_")[1])

        expected_kernel = kernel_name
        expected_csvrow = csvrow_idx

        json_kernel = json_obj["kernel_name"]
        json_target = json_obj["target"]
        # The json row index per-kernel is encoded implicitly by line order;
        # we can’t recover it directly, but we expect kernel_name to match.
        print(f"\n[i={i}]")
        print("  Dataset gname:", g.gname)
        print("  Dataset kernel (normalized):", expected_kernel)
        print("  Dataset csvrow:", expected_csvrow)
        print("  JSON kernel_name:", json_kernel)
        print("  JSON target:", json_target)
        print("  Embedding norm:", embs[i].norm().item())

        if json_kernel != expected_kernel:
            print("  [WARN] Kernel mismatch at index", i)

if __name__ == "__main__":
    main()

