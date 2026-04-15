import json
from collections import defaultdict
from pathlib import Path

from data import MyOwnDataset


def build_json_index(jsonl_path):
    """
    Build a dict: key -> json_obj
    where key = f"{kernel_name}|csvrow_{i}"
    and i is the row index for that kernel in the JSONL (0-based).
    """
    json_map = {}
    per_kernel_count = defaultdict(int)

    with open(jsonl_path, "r") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            kernel_name = obj["kernel_name"]
            i = per_kernel_count[kernel_name]
            per_kernel_count[kernel_name] += 1

            key = f"{kernel_name}|csvrow_{i}"
            # Optional: check for accidental duplicates
            if key in json_map:
                print(f"[WARN] Duplicate key {key} at JSON line {line_no}")
            json_map[key] = obj

    print(f"Loaded {len(json_map)} JSONL entries.")
    return json_map


def filter_jsonl(jsonl_in, jsonl_out, good_files_txt="good_files.txt"):
    # Build the SAME dataset used for GNN & embeddings
    try:
        with open(good_files_txt) as f:
            good_files = [line.strip() for line in f if line.strip()]
        dataset = MyOwnDataset(data_files=good_files)
        print(f"Using filtered dataset with {len(dataset)} graphs")
    except FileNotFoundError:
        dataset = MyOwnDataset()
        print(f"Using full dataset with {len(dataset)} graphs")

    # Load JSON index
    json_map = build_json_index(jsonl_in)

    # For each graph in dataset order, look up the JSON by key
    missing = 0
    matched = 0

    with open(jsonl_out, "w") as fout:
        for idx, g in enumerate(dataset):
            kernel_name = g.kernel  
            suffix = "_processed_result"
            if kernel_name.endswith(suffix):
                kernel_name = kernel_name[:-len(suffix)]
            raw = g.key if isinstance(g.key, str) else str(g.key)
            csvrow = int(raw.split("csvrow_")[1])
            key = f"{kernel_name}|csvrow_{csvrow}"
            obj = json_map.get(key, None)

            if obj is None:
                missing += 1
                print(f"[WARN] No JSON entry for graph idx={idx}, key={key}")
                continue

            matched += 1
            print(f"FOUND idx={idx}, key={key}")
            fout.write(json.dumps(obj) + "\n")

    print(f"Matched {matched} dataset graphs to JSONL rows.")
    print(f"Missing {missing} graphs (no JSON match).")
    print(f"Filtered JSONL written to: {jsonl_out}")


if __name__ == "__main__":
    jsonl_in = "/home/elvouvali/LLM_data/all_kernels_llm_data_multi_target.jsonl"   
    jsonl_out = "/home/elvouvali/LLM_data/all_kernels_llm_data_multi_target_filtered.jsonl"

    filter_jsonl(jsonl_in, jsonl_out)
