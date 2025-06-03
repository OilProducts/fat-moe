import itertools
import os
from typing import Dict, List, Any

import torch, torch.nn as nn, torch.nn.functional as F
from torch import optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tqdm import tqdm
import aim

from abstractinator import ByteLevelTokenizer, ByteSegmentCompressor, CodeExpander, HierarchicalAutoencoder

# ── model pieces from previous messages ────────────────────────────────
#   • ByteSegmentCompressor
#   • CodeExpander
#   (paste their definitions here – unchanged)

N_CPU = int(os.cpu_count()/2)
CACHE_FILE  = "cache/wiki103_bytes.pt"          # optional extra cache


# ── training knobs ────────────────────────────────────────────────────
SEQ_LEN        = 512           # bytes per training chunk
BATCH_SIZE     = 64
TOTAL_STEPS    = 20_000
LR             = 3e-4
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

DIM            = 256           # shared dim per layer
HEADS          = 8
WINDOW         = 128
NUM_QUERIES    = 1
CODEBOOK_L0    = 4096
MULTIPLIER     = 4             # L1 code‑book = 4× larger
CODEBOOK_L1    = CODEBOOK_L0 * MULTIPLIER
BETA           = 0.25


# Example configuration (replace with your actual experiment config)
exp_config = {
    "run_name": "HierarchicalAE_TestRun_v1",
    "project_name": "TemporalAutoencodedLanguageModelling", # Optional: Your AIM project
    "num_levels": 2,
    "initial_vocab_size": 259,
    "compressor_level_configs": [
        {"dim": 256, "heads": 8, "window": 128, "num_queries": 4, "codebook_size": 512, "beta": 0.25},
        {"dim": 256, "heads": 8, "window": 64,  "num_queries": 2, "codebook_size": 1024, "beta": 0.25}
    ],
    "expander_dim_scale": 1.0, "expander_num_enc_layers": 3, "expander_num_dec_layers": 3,
    "expander_heads_scale": 1.0, "expander_dropout": 0.1, "expander_eos_id": 1,
    "propagate_key_padding_mask": True, # Set based on ByteSegmentCompressor modification
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 10,
    "log_interval": 10, # Log metrics to AIM every N steps
}

# --- AIM Setup ---
aim_run = aim.Run(experiment=exp_config.get("run_name", None))
if exp_config.get("project_name"):
    aim_run.repo = f"./aim_repo/{exp_config['project_name']}" # Or your central AIM repo path

# aim_run.track_hparams(exp_config)
for k, v in exp_config.items():
    if k not in ["run_name", "sample_prompt", "sample_tokens"]:
        aim_run[k] = v

# --- Model, Optimizer, DataLoader ---
model = HierarchicalAutoencoder(
    num_levels=exp_config["num_levels"],
    compressor_level_configs=exp_config["compressor_level_configs"],
    initial_vocab_size=exp_config["initial_vocab_size"],
    expander_dim_scale=exp_config["expander_dim_scale"],
    expander_num_enc_layers=exp_config["expander_num_enc_layers"],
    expander_num_dec_layers=exp_config["expander_num_dec_layers"],
    expander_heads_scale=exp_config["expander_heads_scale"],
    expander_dropout=exp_config["expander_dropout"],
    expander_eos_id=exp_config["expander_eos_id"],
    propagate_key_padding_mask=exp_config["propagate_key_padding_mask"]
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=exp_config["learning_rate"])

# # Load raw WikiText‑103 training split (≈103 M tokens)
# wiki = load_dataset("wikitext", "wikitext-103-v1", split="train")
# print(f"Loaded {wiki.num_rows} rows, {wiki.num_rows * 512} tokens")

# Simple byte‑level tokenizer: utf‑8 bytes + BOS/EOS
tokenizer = ByteLevelTokenizer(add_bos=False, add_eos=True)   # defined earlier

try:
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:1%]") # Take first 1% for example
    print(f"Loaded {len(dataset)} examples.")
    print("Dataset features:", dataset.features)
    TEXT_COLUMN_NAME = "text"
except Exception as e:
    print(f"Failed to load dataset: {e}")
    print("Please ensure you have internet access and the dataset name/split is correct.")
    print("Exiting.")
    exit()

def tokenize_examples(examples: Dict[str, List[Any]]) -> Dict[str, List[torch.Tensor]]:
    """
    Tokenizes a batch of text examples.
    'examples' is a dictionary like {'text': ['text1', 'text2', ...]}
    """
    tokenized_inputs = []
    if TEXT_COLUMN_NAME not in examples:
        raise KeyError(f"Expected column '{TEXT_COLUMN_NAME}' not found in dataset examples. Available columns: {list(examples.keys())}")

    for text_content in examples[TEXT_COLUMN_NAME]:
        if text_content is None: # Handle potential None texts if any
            tokenized_inputs.append(torch.tensor([tokenizer.eos_id], dtype=torch.long)) # Empty or EOS only
        else:
            tokenized_inputs.append(tokenizer.encode(text_content))
    return {"input_ids": tokenized_inputs}

print("Tokenizing dataset...")
# Using batched=True for efficiency with the map function
# num_proc can be set to the number of CPU cores for parallelization
# For very large datasets, this will create a cache file.
tokenized_dataset = dataset.map(
    tokenize_examples,
    batched=True,
    num_proc=4, # Adjust based on your CPU cores
    remove_columns=[TEXT_COLUMN_NAME] # Remove original text to save space
)
print("Tokenization complete.")
print("First tokenized example input_ids:", tokenized_dataset[0]["input_ids"][:50]) # Print first 50 tokens
print("Length of first tokenized example:", len(tokenized_dataset[0]["input_ids"]))

class CustomCollatorWithPadding:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        batch is a list of dictionaries, e.g., [{'input_ids': tensor1}, {'input_ids': tensor2}, ...]
        """
        # Extract 'input_ids' from each item in the batch
        sequences = [item['input_ids'] for item in batch]

        # Pad sequences to the max length in this batch
        # pad_sequence expects a list of Tensors
        padded_sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        # padded_sequences shape: (batch_size, max_len_in_this_batch)

        # Create key_padding_mask: True for padding, False for actual tokens
        key_padding_mask = (padded_sequences == self.pad_token_id)

        return {
            "input_ids": padded_sequences,
            "key_padding_mask": key_padding_mask
        }

# --- 7. Create the DataLoader ---
collator = CustomCollatorWithPadding(pad_token_id=tokenizer.pad_id)
BATCH_SIZE = 4 # Example batch size, adjust as needed for your memory

train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collator,
    shuffle=True # Shuffle for training
)
print(f"DataLoader created with batch size {BATCH_SIZE}.")

# --- 8. Show an example batch ---
print("\n--- Example Batch ---")
try:
    example_batch = next(iter(train_dataloader))
    tokens_batch = example_batch["input_ids"]
    kpm_batch = example_batch["key_padding_mask"]

    print("Tokens batch shape:", tokens_batch.shape)
    print("KPM batch shape:", kpm_batch.shape)
    print("\nFirst sequence in batch (tokens):")
    print(tokens_batch[0][:100]) # Print first 100 tokens of the first sequence
    print("\nFirst sequence in batch (KPM):")
    print(kpm_batch[0][:100]) # Print first 100 mask values

    print(f"\nPAD token ID used for padding: {tokenizer.pad_id}")
    print(f"Is PAD token present where KPM is True for first token? (Should be True if padded):")
    if tokens_batch.shape[1] > 0 : # If sequence length > 0
      first_token_first_seq = tokens_batch[0, 0]
      first_kpm_first_seq = kpm_batch[0, 0]
      # Find a padded position if one exists
      padded_indices = (kpm_batch[0] == True).nonzero(as_tuple=True)[0]
      if len(padded_indices) > 0:
          idx_to_check = padded_indices[0]
          print(f"At index {idx_to_check}, Token: {tokens_batch[0, idx_to_check]}, KPM: {kpm_batch[0, idx_to_check]}")
      else:
          print("No padding in the first sequence of this batch.")
    else:
        print("Batch sequence length is 0.")


    # Now, `tokens_batch` and `kpm_batch` can be passed to your model:
    # model.train()
    # output = model(tokens_batch.to(device), key_padding_mask=kpm_batch.to(device))
    # ...
except StopIteration:
    print("DataLoader is empty. This might happen if the dataset slice was too small or all data was filtered.")
except Exception as e:
    print(f"Error fetching batch from DataLoader: {e}")
    import traceback
    traceback.print_exc()

#
# if os.path.exists(CACHE_FILE):
#     print("▶ loading tokenised stream from", CACHE_FILE)
#     byte_stream = torch.load(CACHE_FILE, map_location="cpu")
#
# else:
#     # (a) load raw dataset (arrow file is streamed, not all into RAM)
#     wiki = load_dataset("wikitext", "wikitext-103-v1", split="train")
#
#     # (b) parallel, batched tokenisation; map() writes an arrow cache
#     def encode_batch(batch):
#         ids = [tok.encode(t) for t in batch["text"]]        # list[list[int]]
#         return {"ids": [tok.encode(t) for t in batch["text"]]}
#
#
#     wiki = wiki.map(
#         encode_batch,
#         batched=True,
#         num_proc=N_CPU,        # one process per CPU core
#         remove_columns=["text"],
#         desc="⏳ tokenising & caching",
#     )
#
#     # (c) flatten every cached arrow chunk into one long Tensor *once*
#     byte_stream = torch.tensor(
#         list(itertools.chain.from_iterable(wiki["ids"])),
#     )
#     # (d) (optional) torch‑save for next start‑up → ~1‑second reload
#     os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
#     torch.save(byte_stream, CACHE_FILE)
#     print("✔ wrote", CACHE_FILE)
#
#
# total_chunks = byte_stream.numel() // SEQ_LEN
# byte_stream  = byte_stream[: total_chunks * SEQ_LEN].view(total_chunks, SEQ_LEN)
#
# # ---------------------------------------------------------------------
# # 3.  DataLoader – nothing else changes
# # ---------------------------------------------------------------------
# loader = DataLoader(
#     byte_stream,
#     batch_size=BATCH_SIZE,
#     drop_last=True,
#     pin_memory=True,  # nice speed‑up when moving to GPU
# )
#
# print(f"DataLoader: {len(loader)} batches, {BATCH_SIZE}×{SEQ_LEN} bytes")

# # Map each line → tensor of ints, concatenate, chunk into SEQ_LEN
# byte_stream = torch.cat([tok.encode(t).to(torch.int) for t in wiki["text"]])
# total_chunks = byte_stream.numel() // SEQ_LEN
# byte_stream = byte_stream[: total_chunks * SEQ_LEN].view(total_chunks, SEQ_LEN)
#
# loader = DataLoader(byte_stream,
#                     batch_size=BATCH_SIZE,
#                     shuffle=True,
#                     drop_last=True)
# print(f"DataLoader: {len(loader)} batches, {BATCH_SIZE} bytes each")

# Level‑0   bytes  →  L0 codes   (K0 = CODEBOOK_L0)
# comp0 = ByteSegmentCompressor(vocab_size=259,
#                               dim=DIM, heads=HEADS, window=WINDOW,
#                               num_queries=NUM_QUERIES,
#                               codebook_size=CODEBOOK_L0, beta=BETA)
#
# # Level‑1   L0 codes  →  L1 codes  (K1 = CODEBOOK_L1)
# comp1 = ByteSegmentCompressor(vocab_size=CODEBOOK_L0,
#                               dim=DIM, heads=HEADS, window=WINDOW,
#                               num_queries=NUM_QUERIES,
#                               codebook_size=CODEBOOK_L1, beta=BETA)
#
# # Expander  L1 codes  →  reconstructed L0 codes
# expander1 = CodeExpander(K_hi=CODEBOOK_L1,
#                         K_lo=CODEBOOK_L0,
#                         D=DIM, H=HEADS)
#
# expander0 = CodeExpander(K_hi=CODEBOOK_L0,
#                         K_lo=259,  # 259 = byte vocab size
#                         D=DIM, H=HEADS)
#
# model_modules = nn.ModuleList([comp0, comp1, expander1, expander0]).to(DEVICE)
# opt = AdamW(model_modules.parameters(), lr=LR)


# global_step = 0
# pbar = tqdm(total=TOTAL_STEPS, desc="train")
#
# while global_step < TOTAL_STEPS:
#     for batch in loader:
#         if global_step >= TOTAL_STEPS:
#             break
#         batch = batch.to(DEVICE)
#
#         # forward pass
#         out0 = comp0(batch)                  # bytes ➜ L0
#         # out1 = comp1(out0["codes"])          # L0   ➜ L1
#         # exp1   = expander1(out1["codes"],      # teacher forcing
#         #                  out0["codes"])
#         exp0   = expander0(out0["codes"], batch)
#
#
#
#         next_token_loss = F.cross_entropy(exp0["logits"][:,1:,:].permute(0,2,1), batch[:,1:])
#
#
#
#         # reconstruction loss (predict L0 codes)
#         # recon = F.cross_entropy(
#         #     exp0["logits"].view(-1, CODEBOOK_L0),
#         #     out0["codes"].view(-1),
#         #     reduction="mean"
#         # )
#
#         vq_loss = out0["vq_loss"] + out0["vq_loss"]
#         loss = next_token_loss + vq_loss
#
#         # optimise
#         opt.zero_grad(set_to_none=True)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(comp0.parameters(), 1.0)
#         torch.nn.utils.clip_grad_norm_(expander0.parameters(), 1.0)
#         opt.step()
#
#         # logging
#         pbar.set_postfix(loss=float(loss),
#                          next_tok=float(next_token_loss),
#                          vq=float(vq_loss))
#         global_step += 1
#         pbar.update(1)
# pbar.close()


# --- Training Loop ---
global_step = 0
model.train()

for epoch in range(exp_config["num_epochs"]):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{exp_config['num_epochs']}")
    for batch_idx, (tokens, kpm) in enumerate(progress_bar):
        tokens = tokens.to(DEVICE)
        kpm = kpm.to(DEVICE) if exp_config["propagate_key_padding_mask"] else None

        optimizer.zero_grad()
        output = model(tokens, key_padding_mask=kpm)

        total_loss = output['total_loss']
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Optional
        optimizer.step()

        if global_step % exp_config["log_interval"] == 0:
            # --- AIM Logging ---
            aim_run.track(total_loss.item(), name='loss.total', step=global_step, epoch=epoch,
                          context={"subset": "train"})
            aim_run.track(output['vq_loss'].item(), name='loss.vq', step=global_step, epoch=epoch,
                          context={"subset": "train"})
            aim_run.track(output['avg_reconstruction_loss'].item(), name='loss.reconstruction_avg', step=global_step,
                          epoch=epoch, context={"subset": "train"})

            for name, val in output['reconstruction_loss_details'].items():
                aim_run.track(val.item(), name=f'loss_detail.{name}', step=global_step, epoch=epoch,
                              context={"subset": "train"})

            for i, ratio in enumerate(output['compression_ratios']):
                aim_run.track(ratio, name=f'compression.ratio_L{i}', step=global_step, epoch=epoch,
                              context={"subset": "train"})
            for i, length in enumerate(output['input_seq_lengths_compressors']):
                aim_run.track(length, name=f'compression.input_len_L{i}', step=global_step, epoch=epoch,
                              context={"subset": "train"})
            for i, length in enumerate(output['output_seq_lengths_compressors']):
                aim_run.track(length, name=f'compression.output_len_L{i}', step=global_step, epoch=epoch,
                              context={"subset": "train"})

            # You can log more things like learning rate, grad norms, etc.
            # aim_run.track(optimizer.param_groups[0]['lr'], name='lr', step=global_step, epoch=epoch)

            progress_bar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "vq": f"{output['vq_loss'].item():.4f}",
                "ratios": ", ".join([f"{r:.2f}" for r in output['compression_ratios']])
            })

        global_step += 1

print("Training finished.")