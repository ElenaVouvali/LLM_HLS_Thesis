from data import MyOwnDataset
from torch_geometric.data import Batch
import torch

ds = MyOwnDataset()
files = ds.processed_file_names  # list of .pt paths

good_files = []
bad_files = []

ref = None
for idx, path in enumerate(files):
    print(idx, "->", path)

    # 1) robust load
    try:
        g = torch.load(path, weights_only=False)
    except Exception as e:
        print(f"[idx={idx}] FAILED TO LOAD {path}: {e}")
        bad_files.append(path)
        continue

    # 2) remove non-tensor attribute that breaks batching
    if hasattr(g, "edge_id_to_idx"):
        del g.edge_id_to_idx

    # 3) establish reference graph
    if ref is None:
        ref = g
        good_files.append(path)
        continue

    # 4) check if this graph can batch with the reference
    try:
        Batch.from_data_list([ref, g])
        good_files.append(path)
    except Exception as e:
        print(f"[idx={idx}] BAD graph at {path} ->", e)
        bad_files.append(path)

print("Total graphs:", len(files))
print("Good graphs:", len(good_files))
print("Bad graphs:", len(bad_files))

with open("good_files.txt", "w") as f:
    for p in good_files:
        f.write(p + "\n")

with open("bad_files.txt", "w") as f:
    for p in bad_files:
        f.write(p + "\n")

