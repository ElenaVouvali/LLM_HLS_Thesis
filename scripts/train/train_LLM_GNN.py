#-----------------------------------------------------------
#                     Imports And Set GPU
#-----------------------------------------------------------

import re
import os
import random
import numpy as np
import torch
import tempfile
import uuid
import shutil

from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from cpp_to_gnn_emb import graph_emb_from_cpp
from collections import Counter
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch.nn.functional as F
# import wandb

GPU_ID = 0
torch.cuda.set_device(GPU_ID)
print("Using CUDA device:", GPU_ID, torch.cuda.get_device_name(GPU_ID))



#-----------------------------------------------------------
#            Load Dataset And Kernel-Family-Level Splitting
#-----------------------------------------------------------

JSONL_PATH = "/home/ubuntu/LLM_data/all_kernels_llm_data_lift_filtered.jsonl"

dataset_dict = load_dataset("json", data_files={"raw": JSONL_PATH})
ds_all = dataset_dict["raw"]

# Find easily in the json file the example
ds_all = ds_all.add_column("json_id", list(range(len(ds_all))))

def family_id_from_kernel_name(name: str) -> str:
    s = name.strip().replace("-", "_")  # normalize

    # MachSuite: treat each kernel as its own family (8 families)
    if s.startswith("machsuite_"):
        return name  # keep original label

    # SPCL: one family
    if s.startswith("spcl_example"):
        return "spcl_example"

    # Serrano: one family
    if s.startswith("serrano_") or s.startswith("serrano-"):
        return "serrano_kalman_filter"

    # Rodinia: group variants into algorithm families
    if s.startswith("rodinia_"):
        rest = s[len("rodinia_"):]  
        if rest.startswith("cfd_flux"):
            return "rodinia_cfd_flux"
        if rest.startswith("cfd_step_factor"):
            return "rodinia_cfd_step_factor"
        if rest.startswith("lc_gicov"):
            return "rodinia_lc_gicov"
        if rest.startswith("lc_mgvf"):
            return "rodinia_lc_mgvf"

        algo = rest.split("_")[0]  
        return f"rodinia_{algo}"

    # fallback: itself
    return s


# Ensure weight is float and add family field
ds_all = ds_all.map(lambda ex: {
    "weight": float(ex["weight"]),
    "family": family_id_from_kernel_name(ex["kernel_name"])
}, num_proc=4)


# Get list of kernels and families
kernel_names = sorted(set(ds_all["kernel_name"]))
families = sorted(set(ds_all["family"]))
print("Total kernels:", len(kernel_names))
print("Kernel names:", kernel_names)
print("Total families:", len(families))
print("Family names:", families)

# train / test / val split
VAL_FAMS  = {"rodinia_pathfinder", "machsuite-sort-radix"}
TEST_FAMS = {"serrano_kalman_filter"}
TRAIN_FAMS = set(families) - VAL_FAMS - TEST_FAMS

print("Train families ({}):".format(len(TRAIN_FAMS)), sorted(TRAIN_FAMS))
print("Val families   ({}):".format(len(VAL_FAMS)),  sorted(VAL_FAMS))
print("Test families  ({}):".format(len(TEST_FAMS)), sorted(TEST_FAMS))

train_fams = sorted(list(TRAIN_FAMS))
val_fams   = sorted(list(VAL_FAMS))
test_fams  = sorted(list(TEST_FAMS))

# Ensures that each split will include all the design points from FAMS respectively and nothing else
def in_family(example, names):
    return example["family"] in names

def resample_train_split_by_weight(ds_train, threshold=0.5, high_repeat=4, low_keep_prob=0.20, seed=123):
    """
    LIFT-like resampling:
      - oversample high-weight (low-latency) points
      - keep only a fraction of low-weight points
      - still preserve some bad / weak designs so the model learns what to avoid
    """
    rng = random.Random(seed)
    selected_indices = []

    weights = ds_train["weight"]
    for i, w in enumerate(weights):
        w = float(w)
        if w >= threshold:
            selected_indices.extend([i] * high_repeat)
        else:
            if rng.random() < low_keep_prob:
                selected_indices.append(i)

    rng.shuffle(selected_indices)
    rows = [ds_train[int(i)] for i in selected_indices]
    out = Dataset.from_list(rows)

    print(
        f"[RESAMPLE] original_train={len(ds_train)} "
        f"resampled_train={len(out)} "
        f"threshold={threshold} high_repeat={high_repeat} low_keep_prob={low_keep_prob}"
    )
    return out


# Filter dataset 
ds_train = ds_all.filter(in_family, fn_kwargs={"names": train_fams}, num_proc=4)
ds_val   = ds_all.filter(in_family, fn_kwargs={"names": val_fams}, num_proc=4)
ds_test  = ds_all.filter(in_family, fn_kwargs={"names": test_fams}, num_proc=4)

ds_train = ds_train.add_column("row_id", list(range(len(ds_train))))
ds_val   = ds_val.add_column("row_id", list(range(len(ds_val))))
ds_test  = ds_test.add_column("row_id", list(range(len(ds_test))))

ds_train = resample_train_split_by_weight(
    ds_train,
    threshold=0.5,
    high_repeat=4,
    low_keep_prob=0.20,
    seed=123,
)

print(ds_train)
print(ds_val)
print(ds_test)

ds = DatasetDict(
    train=ds_train,
    validation=ds_val,
    test=ds_test,
    )



#-----------------------------------------------------------
#               Prompt Format And Tokenization
#-----------------------------------------------------------

MODEL_NAME = "deepseek-ai/deepseek-coder-7b-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


gnn_embeddings = torch.load("/home/ubuntu/GNN_embeddings/all_kernels_gnn_embeddings.pt")
# gnn_embeddings = gnn_embeddings.to(f"cuda:{GPU_ID}")


PROMPT_TEMPLATE = """
### Task
Given this HLS kernel with pragma placeholders of the form auto{{...}}, predict pragma values that minimize latency.

Output ONLY the filled directives, one per line, using EXACTLY the placeholders that appear in the kernel.

Use exactly this format:
auto{{_PIPE_LX}} = <non-negative integer>
auto{{_UNROLL_LX}} = <non-negative integer>
auto{{_ARRAY_T_LX}} = <block|cyclic|complete>
auto{{_ARRAY_F_LX}} = <non-negative integer>
auto{{_ARRAY_D_LX}} = <non-negative integer>

Do not add explanations, comments, or extra headings.

### Kernel
{code}

### Directives:
""".strip()


def format_example(ex):
    prompt = PROMPT_TEMPLATE.format(code=ex["input"])
    full_text = prompt + ex["target"]
    return {"prompt": prompt, "full_text": full_text}


ds_all = ds_all.map(format_example, num_proc=4)
ds = ds.map(format_example, num_proc=4)


MAX_LENGTH = 3072  # context window for Deepseek-coder-7b : 4096
                   # max_length = 3072 to leave room for target tokens

# We want to preserve the end of full_text (which contains all the directive lines)
# If the prompt + target tokens exceed the context window of the model then cut from the beginning of the prompt (maybe early code lines).
# No information regarding the targets is lost

def tokenize_and_mask(ex):
    # Tokenize prompt --> tokenize() returns input_ids, attention_mask => prompt_ids are the token IDs for prompt
    prompt_ids = tokenizer(
        ex["prompt"],
        add_special_tokens=False,
    )["input_ids"]

    # Tokenize target --> target_ids are the token IDs for target
    target_ids = tokenizer(
        ex["target"],
        add_special_tokens=False,
    )["input_ids"]

    t_len = len(target_ids)

    # Build input_ids with manual truncation: the goal is to keep all target_ids if possible.
    if t_len >= MAX_LENGTH: # target alone is too long for the model's window --> not a very realistic case for our data
        input_ids = target_ids[-MAX_LENGTH:]  # keep only the tail
        prompt_len = 0
    else:
        # Room left in window for prompt tokens
        remaining_for_prompt = MAX_LENGTH - t_len

        # Keep only the last "remaining_for_prompt" tokens of the prompt
        if len(prompt_ids) > remaining_for_prompt:
            prompt_ids = prompt_ids[-remaining_for_prompt:]

        input_ids = prompt_ids + target_ids
        prompt_len = len(prompt_ids)

    # Build attention mask and pad to MAX_LENGTH if needed
    attention_mask = [1] * len(input_ids)

    labels = np.array(input_ids, dtype=np.int64)
    labels[:prompt_len] = -100

    # Also we ignore padding, where attention_mask == 0
    attn = np.array(attention_mask, dtype=np.int64)
    labels[attn == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "example_weight": float(ex["weight"]),
        "json_id": int(ex["json_id"]),
        }


cols_to_remove = ds["train"].column_names  # remove raw fields, replace by tokenized
# We keep : input_ids, attention_mask, labels, example_weight etc
# We remove : Raw input, target, kernel_name, prompt etc

# Do not remove the 'idx' column, we need it for the collator
if "json_id" in cols_to_remove:
    cols_to_remove.remove("json_id")

tokenized_ds = ds.map(
    tokenize_and_mask,
    batched=False,
    remove_columns=cols_to_remove,
    desc="Tokenizing",
)

tokenized_ds


#-----------------------------------------------------------
#                     Data Collator
#-----------------------------------------------------------

def data_collator(features):
    max_len = max(len(f["input_ids"]) for f in features)

    def pad_1d(x, pad_value):
        return x + [pad_value] * (max_len - len(x))

    batch = {}
    batch["input_ids"] = torch.tensor(
        [pad_1d(f["input_ids"], tokenizer.pad_token_id) for f in features],
        dtype=torch.long,
    )
    batch["attention_mask"] = torch.tensor(
        [pad_1d(f["attention_mask"], 0) for f in features],
        dtype=torch.long,
    )
    batch["labels"] = torch.tensor(
        [pad_1d(f["labels"].tolist() if isinstance(f["labels"], np.ndarray) else f["labels"], -100) for f in features],
        dtype=torch.long,
    )
    batch["example_weight"] = torch.tensor(
        [float(f["example_weight"]) for f in features],
        dtype=torch.float32,
    )
    batch["json_id"] = torch.tensor(
        [f["json_id"] for f in features],
        dtype=torch.long,
    )
    batch["gnn_emb"] = gnn_embeddings[batch["json_id"]]
    return batch


#-----------------------------------------------------------
#       Load Model (in 4-bit) And Attach QLoRA (with PEFT)
#-----------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # reduces GPU memory
    bnb_4bit_use_double_quant=True, # double quantization --> further reduces memory with small quality loss
    bnb_4bit_quant_type="nf4",  # normal float 4 quantization --> preserves model quality better than naive 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    # even though weights are 4-bit, computations are done in bfloat16 or float16 for numerical stability
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": GPU_ID},  # places the model layers on GPU
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

model = prepare_model_for_kbit_training(model)
# marks the right parts of the 4-bit model so gradients can flow through the LoRA adapters + enables gradient checkpointing (use_cache=False)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False

# This keeps the 7B base model frozen in 4-bit and only train the small LoRA adapter matrices
lora_config = LoraConfig(
    r=8, # rank of the low-rank factors
    lora_alpha=16,  # scaling factor --> larger alpha => bigger LoRA influence, too large may destabilize training
    lora_dropout=0.05,  # adds regularization on the adapters to avoid overfitting
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # typical LLaMA-style modules; Deepseek-coder is LLaMA-like
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # all attention projections (g, k, v, o) and feed-forward (MLP) projections (gate, up, down)
    )

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


#-----------------------------------------------------------------
#       Wrapper for Weighted Fine Tuning with GNN Supervision
#-----------------------------------------------------------------

# B = batch_size , T = sequence_length , V = vocab_size

class LIFT_Trainer(Trainer):
    def __init__(self, *args, raw_dataset=None, tokenizer=None, graph_emb_func=None, lambda_gnn=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_dataset = raw_dataset 
        self.tokenizer = tokenizer
        self.graph_emb_func = graph_emb_func
        self.lambda_gnn = lambda_gnn
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        example_weight = inputs.pop("example_weight").to(model.device)
        gnn_target = inputs.pop("gnn_emb").to(model.device) # Ground truth embedding (B, 64)
        batch_indices = inputs.pop("json_id") 
        
        # Forward Pass (get logits)
        outputs = model(**inputs)
        logits = outputs.logits # (B, T, V)

        ########## Cross Entropy Loss (L_CE) ##########
        # token at position t is used to predict token at position t+1 => we map logit at time t with the label at time t+1
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()   # (B, T-1)
        loss_mask = shift_labels.ne(-100).float()   # (B, T-1)

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),   # (B*(T-1), V)
            shift_labels.view(-1),  # (B*(T-1),)
            reduction="none",
            ignore_index=-100
        ).view(shift_labels.size())         # Reshape back to (B, T-1)

        # CE Loss per example (average ce loss over target tokens)
        ce_per_example = (loss_per_token * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0) # (B,)
        # print(f"Cross Entropy Loss : {ce_per_example}")


        ########## Graph Embedding Loss (L_GNN) ##########

        mse_tensor = torch.zeros_like(ce_per_example)

        # greedy decode to get predicted pragmas
        with torch.no_grad(): # No gradients needed for generation
            pred_next_ids = torch.argmax(shift_logits, dim=-1) # (B, T-1)
        
        if self.raw_dataset is not None:
            # Move decoding to CPU to avoid sync overhead
            pred_next_ids_cpu = pred_next_ids.detach().cpu().tolist()
            shift_labels_cpu  = shift_labels.cpu().tolist()
            batch_indices_cpu = batch_indices.detach().cpu().tolist()
            
            # loop over batch to generate graphs
            mse_scores = []

            for i, global_idx in enumerate(batch_indices_cpu):
                # Get the raw C++ template
                raw_example = self.raw_dataset[int(global_idx)] 
                template_code = raw_example['input'] # c++ code with auto-placeholders

                # Decode ONLY the target-token positions (where label != -100)
                mask_i = [label != -100 for label in shift_labels_cpu[i]]
                pred_ids_i = [tok for tok, m in zip(pred_next_ids_cpu[i], mask_i) if m]
                
                pred_text = self.tokenizer.decode(pred_ids_i, skip_special_tokens=True)
                #print(pred_text)
                
                # Fill the placeholders with the predicted values
                raw_assigns = parse_assignments(pred_text)
                resolved_assigns = resolve_assigns_for_template(template_code, raw_assigns)
                #print(template_code)
                #print(resolved_assigns)

                # Get the GNN embedding
                unique_id = str(uuid.uuid4())
                pred_emb = self.graph_emb_func(template_code, resolved_assigns, unique_id)
                pred_emb = pred_emb.to(gnn_target.device)
                # pred_norm = torch.linalg.norm(pred_emb)
                # print(f"Norm of pred_emb: {pred_norm.item():.4f}")
                # gt_norm = torch.linalg.norm(gnn_target[i])
                # print(f"Norm of gt_emb: {gt_norm.item():.4f}")

                # Calculate MSE for this example
                mse = F.mse_loss(pred_emb, gnn_target[i])
                # print(mse)
                mse_scores.append(mse)
            
            # Stack MSE scores into a tensor
            mse_tensor = torch.stack(mse_scores).to(ce_per_example.device) # (B,)
            # print(f"MSE tensor : {mse_tensor}")

        mse_tensor = mse_tensor / mse_tensor.detach().mean().clamp(min=1e-6)
        mse_tensor = mse_tensor.clamp(min=0.0, max=3.0)

        # Soft Structural Supervision : Weight = 1 + lambda * MSE
        # If prediction is bad => High MSE => high Weight
        structural_weight = 1.0 + (self.lambda_gnn * mse_tensor)
        # print(f"Structural Weight : {structural_weight}")

        # Combine: Example Weight (Pareto + Kernel) * Structural Weight * CE Loss
        # We detach structural_weight because we don't backprop through it
        final_weights = example_weight * structural_weight.detach()
        final_weights = final_weights.clamp(min=0.1, max=5.0)
        # print(f"Final Weight : {final_weights}")

        # weighted_loss = (ce_per_example * final_weights).mean()
        eps = 1e-8
        weighted_loss = (ce_per_example * final_weights).sum() / (final_weights.sum() + eps)
        # print(f"Weighted Loss : {weighted_loss}")

        if return_outputs:
            return weighted_loss, outputs
        return weighted_loss


_assign_pat = re.compile(
    r"""
    (?:auto)?          # optional 'auto'
    \{                 # opening brace
    ([_A-Z0-9]{3,20})  # group(1): Key must be uppercase/numbers/underscore, 3-20 chars long
    \}                 # closing brace
    \s*=\s* # '=' with optional spaces
    ([a-zA-Z0-9_]+)    # group(2): value
    """,
    re.VERBOSE,
)

_placeholder_pat = re.compile(r'auto\{([^}]+)\}')


def extract_placeholders(template_code: str) -> list:
    # returns e.g. ["_PIPE_L1", "_UNROLL_L1", "_ARRAY_T_L6", ...]
    return [m.group(1).strip().upper() for m in _placeholder_pat.finditer(template_code)]


def build_fallback_map(raw_assigns: dict) -> dict:
    fallback = {}
    for k, v in raw_assigns.items():
        suffix_match = re.search(r'L\d+', k)
        if not suffix_match:
            continue
        suffix = suffix_match.group(0)

        if "PIPE" in k or "PIROLL" in k:
            fallback[f"PIPE_{suffix}"] = v
        elif "UNROLL" in k:
            fallback[f"UNROLL_{suffix}"] = v
        elif "ARRAY_T" in k:
            fallback[f"ARRAY_T_{suffix}"] = v
        elif "ARRAY_F" in k:
            fallback[f"ARRAY_F_{suffix}"] = v
        elif "ARRAY_D" in k:
            fallback[f"ARRAY_D_{suffix}"] = v
    return fallback


_ALLOWED_PREFIXES = ("_PIPE_", "_UNROLL_", "_ARRAY_T_", "_ARRAY_F_", "_ARRAY_D_")
_ALLOWED_KEY_RE = re.compile(r"^_(PIPE|UNROLL|ARRAY_T|ARRAY_F|ARRAY_D)_L\d+$")


def coerce_value(key: str, value: str) -> str:
    """
    Enforce your typing rules:
    - ARRAY_T: block|cyclic|complete else default '0'
    - others: digits only, else 0 (or 1 for ARRAY_D if you want)
    """
    if key.startswith("_ARRAY_T_"):
        v = str(value).lower().strip()
        if v in ("block", "cyclic", "complete"):
            return v
        # if it's garbage like 'blockic', your normalize_hls_val already maps it to 'block'
        v2 = normalize_hls_val(v)
        return v2 if v2 in ("block", "cyclic", "complete") else "block"
    else:
        v = str(value).strip()
        return v if v.isdigit() else ("1" if key.startswith("_ARRAY_D_") else "0")


def resolve_assigns_for_template(template_code: str, raw_assigns: dict) -> dict:
    # normalize keys and values from raw assigns
    raw = {k.strip().upper(): normalize_hls_val(v) for k, v in raw_assigns.items()}
    fallback = build_fallback_map(raw)

    resolved = {}
    for key in extract_placeholders(template_code):
        key = key.upper()

        # Only resolve keys we consider valid pragma placeholders
        if not _ALLOWED_KEY_RE.match(key):
            # if your template has other placeholder families, handle them here
            continue

        # Priority 1: exact match
        value = raw.get(key)

        # Priority 2: category+suffix fallback
        if value is None:
            suffix_match = re.search(r"L\d+", key)
            if suffix_match:
                suffix = suffix_match.group(0)
                if key.startswith("_PIPE_"):
                    value = fallback.get(f"PIPE_{suffix}")
                elif key.startswith("_UNROLL_"):
                    value = fallback.get(f"UNROLL_{suffix}")
                elif key.startswith("_ARRAY_T_"):
                    value = fallback.get(f"ARRAY_T_{suffix}")
                elif key.startswith("_ARRAY_F_"):
                    value = fallback.get(f"ARRAY_F_{suffix}")
                elif key.startswith("_ARRAY_D_"):
                    value = fallback.get(f"ARRAY_D_{suffix}")

        # Priority 3: default
        if value is None:
            value = "0"

        resolved[key] = coerce_value(key, value)

    return resolved


def normalize_hls_val(val):
    val = str(val).lower().strip()
    if 'complete' in val: return 'complete'
    if 'cycl' in val:     return 'cyclic'
    if 'block' in val:    return 'block'
    # For numeric values, keep only the digits
    numeric_match = re.search(r'\d+', val)
    if numeric_match:
        return str(int(numeric_match.group(0))) # This turns "04" into "4"
    return val


def parse_assignments(text: str):
    found = {}
    for m in _assign_pat.finditer(text):
        key = m.group(1).strip().upper()
        val = normalize_hls_val(m.group(2))
        found[key] = val
    return found


#-----------------------------------------------------------
#                       Training
#-----------------------------------------------------------

OUTPUT_DIR = "/home/ubuntu/deepseek_7B_qLoRA_GNN_all_kernels_perf"

# os.environ["WANDB_WATCH"] = "all"

# wandb.login()

# ORIG_RUN_ID = "lm7c9kez"
# run = wandb.init(
#    project = "llm-for-hls",
#    id = ORIG_RUN_ID, # id of the interrupted run
#    resume = "must",  # resume the interrupted run
#    config = {
#      "learning_rate": 2e-4,
#      "architecture": "deepseek-ai/deepseek-coder-7b-base",
#      "dataset": "all_kernels_llm_data.jsonl",
#      "epochs": 3,
#    }
# )

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # 2
    gradient_accumulation_steps=8,  # effective batch size = 4*2 = 8 (fits in memory + reasonably stable gradients)
    num_train_epochs=2,
    max_steps=500,
    learning_rate=2e-4, # typical for LoRA
    weight_decay=0.0,
    warmup_ratio=0.03,  # 3% of total steps are LR warm-up => we avoid large gradients at the start
    lr_scheduler_type="cosine", # LR gently decays to near-zero
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps = 20,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",

    dataloader_num_workers=10,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    report_to="none",
    remove_unused_columns = False,
)

trainer = LIFT_Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    raw_dataset=ds_all,         
    tokenizer=tokenizer,         
    graph_emb_func=graph_emb_from_cpp, 
    lambda_gnn=3.0  # strength of the GNN supervision
)



# --- SANITY CHECK START ---
# print("\n--- Running One-Batch Sanity Check ---")
# 1. Grab a small batch from the train dataset
# small_eval_batch = [tokenized_ds["train"][i] for i in range(8)]

# 2. Run it through the collator
# batch = data_collator(small_eval_batch)
# print(f"Batch keys: {batch.keys()}")
# print(f"GNN Embedding shape: {batch['gnn_emb'].shape}") # Should be (8, 64)

# 3. Test the Loss Computation Logic
# Move batch to device
# batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# Trigger the compute_loss manually
# model.train()
# try:
#     loss = trainer.compute_loss(model, batch)
#     print(f"Success! Test Loss: {loss.item():.4f}")
# except Exception as e:
#     print(f"Sanity Check Failed! Error: {e}")
     # This will catch if fill_placeholders or graph_emb_func has an issue
# --- SANITY CHECK END ---


# Resume from checkpoint
CHECKPOINT_PATH = "~/deepseek_7B_qLoRA_GNN_all_kernels_perf/checkpoint-500"
if CHECKPOINT_PATH is not None and os.path.isdir(CHECKPOINT_PATH):
    print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
    trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
else:
    print(f"No checkpoint found at {CHECKPOINT_PATH}, starting from scratch.")
    trainer.train()


# Save only the LoRA adapter
trainer.model.save_pretrained(OUTPUT_DIR + "/lora_adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/tokenizer")

# wandb.finish()

