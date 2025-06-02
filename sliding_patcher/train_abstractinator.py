import itertools
import os

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from abstractinator import ByteLevelTokenizer, ByteSegmentCompressor, CodeExpander

# ── model pieces from previous messages ────────────────────────────────
#   • ByteSegmentCompressor
#   • CodeExpander
#   (paste their definitions here – unchanged)

N_CPU = int(os.cpu_count()/2)
CACHE_FILE  = "cache/wiki103_bytes.pt"          # optional extra cache


# ── training knobs ────────────────────────────────────────────────────
SEQ_LEN        = 512           # bytes per training chunk
BATCH_SIZE     = 2
TOTAL_STEPS    = 20_000
LR             = 3e-4
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

DIM            = 256           # shared dim per layer
HEADS          = 8
WINDOW         = 128
NUM_QUERIES    = 1
CODEBOOK_L0    = 1024
MULTIPLIER     = 4             # L1 code‑book = 4× larger
CODEBOOK_L1    = CODEBOOK_L0 * MULTIPLIER
BETA           = 0.25

# Load raw WikiText‑103 training split (≈103 M tokens)
wiki = load_dataset("wikitext", "wikitext-103-v1", split="train")
print(f"Loaded {wiki.num_rows} rows, {wiki.num_rows * 512} tokens")

# Simple byte‑level tokenizer: utf‑8 bytes + BOS/EOS
tok = ByteLevelTokenizer(add_bos=False, add_eos=True)   # defined earlier


if os.path.exists(CACHE_FILE):
    print("▶ loading tokenised stream from", CACHE_FILE)
    byte_stream = torch.load(CACHE_FILE, map_location="cpu")

else:
    # (a) load raw dataset (arrow file is streamed, not all into RAM)
    wiki = load_dataset("wikitext", "wikitext-103-v1", split="train")

    # (b) parallel, batched tokenisation; map() writes an arrow cache
    def encode_batch(batch):
        ids = [tok.encode(t) for t in batch["text"]]        # list[list[int]]
        return {"ids": [tok.encode(t) for t in batch["text"]]}


    wiki = wiki.map(
        encode_batch,
        batched=True,
        num_proc=N_CPU,        # one process per CPU core
        remove_columns=["text"],
        desc="⏳ tokenising & caching",
    )

    # (c) flatten every cached arrow chunk into one long Tensor *once*
    byte_stream = torch.tensor(
        list(itertools.chain.from_iterable(wiki["ids"])),
    )
    # (d) (optional) torch‑save for next start‑up → ~1‑second reload
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    torch.save(byte_stream, CACHE_FILE)
    print("✔ wrote", CACHE_FILE)


total_chunks = byte_stream.numel() // SEQ_LEN
byte_stream  = byte_stream[: total_chunks * SEQ_LEN].view(total_chunks, SEQ_LEN)

# ---------------------------------------------------------------------
# 3.  DataLoader – nothing else changes
# ---------------------------------------------------------------------
loader = DataLoader(
    byte_stream,
    batch_size=BATCH_SIZE,
    drop_last=True,
    pin_memory=True,  # nice speed‑up when moving to GPU
)

print(f"DataLoader: {len(loader)} batches, {BATCH_SIZE}×{SEQ_LEN} bytes")

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
comp0 = ByteSegmentCompressor(vocab_size=259,
                              dim=DIM, heads=HEADS, window=WINDOW,
                              num_queries=NUM_QUERIES,
                              codebook_size=CODEBOOK_L0, beta=BETA)

# Level‑1   L0 codes  →  L1 codes  (K1 = CODEBOOK_L1)
comp1 = ByteSegmentCompressor(vocab_size=CODEBOOK_L0,
                              dim=DIM, heads=HEADS, window=WINDOW,
                              num_queries=NUM_QUERIES,
                              codebook_size=CODEBOOK_L1, beta=BETA)

# Expander  L1 codes  →  reconstructed L0 codes
expander1 = CodeExpander(K_hi=CODEBOOK_L1,
                        K_lo=CODEBOOK_L0,
                        D=DIM, H=HEADS)

expander0 = CodeExpander(K_hi=CODEBOOK_L0,
                        K_lo=259,  # 259 = byte vocab size
                        D=DIM, H=HEADS)

model_modules = nn.ModuleList([comp0, comp1, expander1, expander0]).to(DEVICE)
opt = AdamW(model_modules.parameters(), lr=LR)


global_step = 0
pbar = tqdm(total=TOTAL_STEPS, desc="train")

while global_step < TOTAL_STEPS:
    for batch in loader:
        if global_step >= TOTAL_STEPS:
            break
        batch = batch.to(DEVICE)

        # forward pass
        out0 = comp0(batch)                  # bytes ➜ L0
        # out1 = comp1(out0["codes"])          # L0   ➜ L1
        # exp1   = expander1(out1["codes"],      # teacher forcing
        #                  out0["codes"])
        exp0   = expander0(out0["codes"], batch)



        next_token_loss = F.cross_entropy(exp0["logits"][:,1:,:].permute(0,2,1), batch[:,1:])



        # reconstruction loss (predict L0 codes)
        # recon = F.cross_entropy(
        #     exp0["logits"].view(-1, CODEBOOK_L0),
        #     out0["codes"].view(-1),
        #     reduction="mean"
        # )

        vq_loss = out0["vq_loss"] + out0["vq_loss"]
        loss = next_token_loss + vq_loss

        # optimise
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(comp0.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(expander0.parameters(), 1.0)
        opt.step()

        # logging
        pbar.set_postfix(loss=float(loss),
                         next_tok=float(next_token_loss),
                         vq=float(vq_loss))
        global_step += 1
        pbar.update(1)
pbar.close()
