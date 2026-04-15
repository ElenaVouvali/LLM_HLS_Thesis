import pandas as pd
import json
from insert_placeholders import insert_placeholders
import os
import re



def load_kernel_metadata(kernel_name, application_dataset_root, apl_mapping_root):
    """
    Build:
      - LOOP_MAP:  { csv_col -> loop_label }
      - ARRAY_MAP: { csv_col -> array_label }
      - LOOP_ITERATIONS_MAP: { loop_label -> iteration_count (int or str) }

    Uses:
      - {application_dataset_root}/{kernel_name}/kernel_info.txt
      - {apl_mapping_root}/{kernel_name}.txt
    """
    kernel_dir = os.path.join(application_dataset_root, kernel_name)
    kernel_info_file = os.path.join(kernel_dir, "kernel_info.txt")
    apl_mapping_file = os.path.join(apl_mapping_root, f"{kernel_name}.txt")

    LOOP_MAP = {}
    ARRAY_MAP = {}
    LOOP_ITERATIONS_MAP = {}

    # Load loop iteration counts and identify which Li are loops/arrays 
    with open(kernel_info_file, "r") as f:
        info_lines = [ln.strip() for ln in f if ln.strip()]

    # First line is kernel name; skip it
    for line in info_lines[1:]:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 2:
            continue
        label = parts[0]            
        kind = parts[1].lower()     # "loop" or "array"

        if kind == "loop":
            # Expected format: Li,loop,<int>
            if len(parts) >= 3:
                LOOP_ITERATIONS_MAP[label] = parts[2]
        # For arrays we do not need iteration count here

    # Load CSV <-> Li mapping from ApplicationAPLMapping
    with open(apl_mapping_file, "r") as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            csv_col, label = [p.strip() for p in ln.split(",") if p.strip()]
            # Decide whether this is loop or array by name
            if csv_col.lower().startswith("array"):
                ARRAY_MAP[csv_col] = label
            else:
                LOOP_MAP[csv_col] = label

    return LOOP_MAP, ARRAY_MAP, LOOP_ITERATIONS_MAP



def create_llm_data_json(
    csv_file,
    cpp_file,
    kernel_name,
    application_dataset_root,
    apl_mapping_root,
    output_jsonl,
    append=False,
):
    """
    Each JSONL line:
      {
        "input":  <kernel code with auto{} placeholders>,
        "target": <pragma assignment text>,
        "area": <(FF_% + DSP_% + BRAM_% + LUT_%)/4>,
        "latency": <latency_msec>
        "kernel_name": <kernel identifier>
      }
    """

    # Build mappings from kernel metadata 
    LOOP_MAP, ARRAY_MAP, LOOP_ITERATIONS_MAP = load_kernel_metadata(
        kernel_name,
        application_dataset_root=application_dataset_root,
        apl_mapping_root=apl_mapping_root,
    )

    # Insert auto{} placeholders into the C++ kernel once per kernel 
    placeholder_lines = insert_placeholders(cpp_file)
    input_template = "".join(placeholder_lines).strip()

    # Load the preprocessed CSV (with preprocessed columns Weight, Area, etc.)
    df = pd.read_csv(csv_file)

    # append or overwrite JSONL 
    mode = "a" if (append and os.path.exists(output_jsonl) and os.path.getsize(output_jsonl) > 0) else "w"

    with open(output_jsonl, mode) as out_f:
        for idx, row in df.iterrows():
            pragma_lines = []

            # ========== Decode loop pragmas ==========
            for csv_col, loop_label in LOOP_MAP.items():
                if csv_col not in df.columns:
                    continue

                cell = row[csv_col]
                if pd.isna(cell):
                    continue

                val = str(cell).strip().lower()
                if val == "auto":
                    # No directive chosen for this loop in this design
                    continue

                pipe = 0
                unroll = 0

                if val == "unroll":
                    # "unroll" means full unroll: factor = iteration count
                    iter_count = LOOP_ITERATIONS_MAP.get(loop_label)
                    if iter_count not in (None, ""):
                        unroll = int(str(iter_count))
                elif val.startswith("unroll_"):
                    # explicit unroll factor: "unroll_4"
                    unroll = int(val.split("_", 1)[1])
                elif val.startswith("pipeline_"):
                    # explicit initiation interval: "pipeline_1"
                    pipe = int(val.split("_", 1)[1])
                elif val == "pipeline":
                    pipe = 1

                pragma_lines.append(f"auto{{_PIPE_{loop_label}}} = {pipe}")
                pragma_lines.append(f"auto{{_UNROLL_{loop_label}}} = {unroll}")

            # ========== Decode array pragmas ==========
            for csv_col, array_label in ARRAY_MAP.items():
                if csv_col not in df.columns:
                    continue

                cell = row[csv_col]
                if pd.isna(cell):
                    continue

                val = str(cell).strip().lower()
                if val == "auto":
                    # No directive for this array in this design
                    continue

                parts = val.split("_")
                if len(parts) == 2:
                    # complete partition: "complete_256" 
                    pragma_lines.append(f"auto{{_ARRAY_T_{array_label}}} = complete")
                    pragma_lines.append(f"auto{{_ARRAY_F_{array_label}}} = 0")
                    pragma_lines.append(f"auto{{_ARRAY_D_{array_label}}} = {int(parts[1])}")
                elif len(parts) >= 3:
                    # block or cyclic: "block_4_64" or "cyclic_2_128"
                    pragma_lines.append(f"auto{{_ARRAY_T_{array_label}}} = {parts[0]}")
                    pragma_lines.append(f"auto{{_ARRAY_F_{array_label}}} = {int(parts[1])}")
                    pragma_lines.append(f"auto{{_ARRAY_D_{array_label}}} = {int(parts[2])}")

            # If no pragma lines at all, you can skip or keep it as "no-op" target
            if not pragma_lines:
                continue

            # Extract Latency and Area Columns
            area = float(row["Area"])
            latency = float(row["Latency_msec"])

            json_entry = {
                "input": input_template,
                "target": "\n".join(pragma_lines),
                "latency": latency,
                "area": area,
                "kernel_name": kernel_name,
            }

            out_f.write(json.dumps(json_entry) + "\n")

    print(f"Wrote {len(df)} samples from {os.path.basename(csv_file)} into {output_jsonl}")


application_dataset_root = "/home/elvouvali/Data4LLMPrompting/ApplicationDataset"
apl_mapping_root = "/home/elvouvali/Data4LLMPrompting/ApplicationAPLMapping"
output_jsonl = "/home/elvouvali/LLM_data/all_kernels_llm_data_muti_target.jsonl"

kernel_names = [d for d in os.listdir(application_dataset_root) 
                if os.path.isdir(os.path.join(application_dataset_root, d))]

print(f"Found {len(kernel_names)} kernels. Starting processing...")

for kernel_name in kernel_names:
    csv_file = f"/home/elvouvali/Data4LLMPrompting/preprocessed_CSVS/preprocessed-{kernel_name}.csv"
    cpp_folder = os.path.join(application_dataset_root, kernel_name)
    cpp_files = [f for f in os.listdir(cpp_folder) if f.endswith('.cpp')]
    if not cpp_files:
        print(f"Skipping {kernel_name}: No .cpp file found.")
        continue
    
    cpp_file = os.path.join(cpp_folder, cpp_files[0])
    print(f"Processing kernel: {kernel_name}")

    create_llm_data_json(
        csv_file=csv_file,
        cpp_file=cpp_file,
        kernel_name=kernel_name,
        application_dataset_root=application_dataset_root,
        apl_mapping_root=apl_mapping_root,
        output_jsonl=output_jsonl,
        append=True,  
)

