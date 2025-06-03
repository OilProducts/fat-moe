import os
import math

import torch
from datasets import load_dataset, Dataset  # Import Dataset for the dummy data
from torch import optim
from torch.utils.data import DataLoader
from typing import List, Dict, Any  # For type hinting

import aim
from tqdm import tqdm

from abstractinator import HierarchicalAutoencoder

def short_num(n):
    n = float(n)
    millnames = ['', 'k', 'm', 'b', 't', 'q']

    if n == 0:
        return '0'
    # Determine the appropriate suffix index
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(math.log10(abs(n)) / 3))))
    # Scale the number down by the appropriate power of 1000
    scaled = n / 10 ** (3 * millidx)
    # Determine the number of decimal places based on the scaled value
    if scaled < 10:
        formatted = f"{scaled:.2f}"
    elif scaled < 100:
        formatted = f"{scaled:.1f}"
    else:
        formatted = f"{scaled:.0f}"
    return f"{formatted}{millnames[millidx]}"

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
        {"dim": 256, "heads": 8, "window": 128, "num_queries": 1, "codebook_size": 2048, "beta": 0.25},
        {"dim": 512, "heads": 8, "window": 64,  "num_queries": 1, "codebook_size": 32768, "beta": 0.25}
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

# print number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model initialized with {short_num(num_params)} trainable parameters.")

optimizer = optim.AdamW(model.parameters(), lr=exp_config["learning_rate"])


# Your ByteLevelTokenizer class (ensure torch is imported for torch.tensor)
class ByteLevelTokenizer:
    """
    Minimal Byte-level tokenizer:
      - raw bytes are 0..255
      - special tokens are >= 256
      - optional BOS/EOS
    """

    def __init__(
            self,
            bos_id: int = 256,
            eos_id: int = 257,
            pad_id: int = 258,
            add_bos: bool = True,
            add_eos: bool = True
    ):
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.add_bos = add_bos
        self.add_eos = add_eos
        # Vocabulary size: 256 for bytes + number of special tokens.
        # This assumes special tokens are contiguous and pad_id is the largest.
        self.vocab_size = max(bos_id, eos_id, pad_id) + 1

    def encode(self, text: str, add_bos: bool | None = None, add_eos: bool | None = None) -> torch.Tensor:
        """
        Convert text to a torch.Tensor of integers where each character is its
        raw byte (0-255). Optionally add BOS/EOS tokens outside 0-255 range.
        """
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        raw_bytes = text.encode("utf-8", errors="ignore")
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend(raw_bytes)
        if add_eos:
            tokens.append(self.eos_id)
        return torch.tensor(tokens, dtype=torch.int16)

    def decode(self, tokens: list[int] | torch.Tensor, cut_at_eos: bool = False) -> str:
        """
        Reconstruct text by ignoring any token >= 256 and turning the rest
        (0..255) back into bytes. If `cut_at_eos=True`, stop at the first EOS.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        if cut_at_eos and (self.eos_id in tokens):
            try:  # Add try-except for robustness, though index() should find it if 'in' is true
                eos_index = tokens.index(self.eos_id)
                tokens = tokens[:eos_index]
            except ValueError:
                pass  # Should not happen if check `self.eos_id in tokens` is True

        byte_list = [t for t in tokens if t >= 0 and t < 256]
        return bytes(byte_list).decode("utf-8", errors="ignore")


# Tokenization and processing function to be used with dataset.map()
def tokenize_and_process_examples(
        examples: Dict[str, List[str]],  # Type hint for clarity
        tokenizer: ByteLevelTokenizer,
        sequence_length: int,
        text_column="text"
) -> Dict[str, List[torch.Tensor]]:  # Return type hint
    processed_input_ids_list = []
    processed_labels_list = []
    processed_kpm_list = []  # <<< MODIFICATION: New list for key padding masks

    for text_content in examples[text_column]:
        if not isinstance(text_content, str):
            text_content = str(text_content) if text_content is not None else ""

        encoded_tokens = tokenizer.encode(text_content)
        current_length = len(encoded_tokens)
        processed_single_input_ids: torch.Tensor  # Type hint

        if current_length > sequence_length:
            if tokenizer.add_eos:
                if sequence_length > 0:
                    final_chunk = encoded_tokens[:sequence_length - 1]
                    processed_single_input_ids = torch.cat(
                        (final_chunk, torch.tensor([tokenizer.eos_id], dtype=torch.int16))
                    )
                else:
                    processed_single_input_ids = torch.tensor([], dtype=torch.int16)
            else:
                processed_single_input_ids = encoded_tokens[:sequence_length]
        elif current_length < sequence_length:
            if sequence_length > 0:
                padding_needed = sequence_length - current_length
                padding_tensor = torch.full((padding_needed,), tokenizer.pad_id, dtype=torch.int16)
                processed_single_input_ids = torch.cat((encoded_tokens, padding_tensor))
            else:
                processed_single_input_ids = torch.tensor([], dtype=torch.int16)
        else:
            processed_single_input_ids = encoded_tokens

        # Safeguard: Ensure final tensor is exactly sequence_length if sequence_length > 0
        # This step is crucial if the above logic might miss an edge case for exact length.
        if sequence_length > 0 and len(processed_single_input_ids) != sequence_length:
            # This path indicates an issue in the primary truncation/padding logic if hit.
            # For robustness, we explicitly resize and pad/truncate if necessary.
            if len(processed_single_input_ids) > sequence_length:
                processed_single_input_ids = processed_single_input_ids[:sequence_length]
            else:  # len < sequence_length
                padding_needed = sequence_length - len(processed_single_input_ids)
                padding_tensor = torch.full((padding_needed,), tokenizer.pad_id, dtype=torch.int16)
                processed_single_input_ids = torch.cat((processed_single_input_ids, padding_tensor))

        # <<< MODIFICATION: Generate key_padding_mask >>>
        # True where padded (i.e., token is pad_id), False otherwise.
        # This is done *after* padding/truncation to ensure the mask matches the final sequence.
        single_kpm = (processed_single_input_ids == tokenizer.pad_id)

        processed_input_ids_list.append(processed_single_input_ids)
        processed_labels_list.append(processed_single_input_ids.clone())
        processed_kpm_list.append(single_kpm)  # <<< MODIFICATION: Add the kpm

    return {
        "input_ids": processed_input_ids_list,
        "labels": processed_labels_list,
        "key_padding_mask": processed_kpm_list  # <<< MODIFICATION: Return the list of KPMs
    }


# --- Main script ---

# ⚙️ Initialize your tokenizer
tokenizer = ByteLevelTokenizer(add_bos=True, add_eos=True)
print(f"Tokenizer initialized. BOS ID: {tokenizer.bos_id}, EOS ID: {tokenizer.eos_id}, PAD ID: {tokenizer.pad_id}")
print(f"Effective vocabulary size (including special tokens): {tokenizer.vocab_size}")

# 📚 Define parameters
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
TEXT_COLUMN = "text"
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 4

# 1. Load the dataset
print(f"\nLoading dataset '{DATASET_NAME}' with configuration '{DATASET_CONFIG}'...")
raw_dataset = None
try:
    raw_dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train")
    # raw_dataset = raw_dataset.select(range(1000)) # Optional: for faster testing
    print(f"Dataset loaded. Number of examples: {len(raw_dataset)}")
except Exception as e:
    print(f"Error loading dataset '{DATASET_NAME}': {e}")
    print("Using a small dummy dataset for demonstration purposes.")
    dummy_data = {
        "text": [
                    "This is a sample financial news article about stocks.",
                    "Another document discusses market trends and analysis.",
                    "The Federal Reserve announced new monetary policies.",
                    "Understanding corporate earnings reports is crucial for investors.",
                ] * 25
    }
    raw_dataset = Dataset.from_dict(dummy_data)
    TEXT_COLUMN = "text"
    print(f"Dummy dataset created with {len(raw_dataset)} examples.")

# 2. Tokenize and process the dataset
print(f"\nTokenizing and processing dataset with sequence length {SEQUENCE_LENGTH}...")
tokenized_dataset = raw_dataset.map(
    tokenize_and_process_examples,
    batched=True,
    fn_kwargs={
        "tokenizer": tokenizer,
        "sequence_length": SEQUENCE_LENGTH,
        "text_column": TEXT_COLUMN
    },
    remove_columns=raw_dataset.column_names
)
print("Tokenization complete.")

# 3. Set dataset format to PyTorch tensors
print("\nSetting dataset format to PyTorch tensors...")
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "labels", "key_padding_mask"]  # <<< MODIFICATION: Add "key_padding_mask"
)
print("Dataset format set.")

# 4. Create PyTorch DataLoader
print(f"\nCreating PyTorch DataLoader with batch size {BATCH_SIZE}...")
train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
print("DataLoader created.")

# --- Example Usage ---
print("\n✅ Setup complete. 'train_dataloader' is ready for your training loop.")
print("\n🔎 Example of a batch from the DataLoader:")
try:
    # --- Training Loop ---
    global_step = 0
    model.train()

    for epoch in range(exp_config["num_epochs"]):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{exp_config['num_epochs']}")
        for batch_idx, batch in enumerate(progress_bar):
            tokens = batch["input_ids"]
            kpm = batch["key_padding_mask"]
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
                aim_run.track(output['avg_reconstruction_loss'].item(), name='loss.reconstruction_avg',
                              step=global_step,
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

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current state...")
    # Save the model state
    torch.save(model.state_dict(), "model_checkpoint.pth")