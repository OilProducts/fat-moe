"""Tiny training harness for MoETransformerLM on the streaming `fineweb-edu` dataset.

The goal is **not** to train a state‑of‑the‑art model here – just to give you a
reproducible, end‑to‑end pipeline you can tweak.

Usage (single‑GPU example):

```bash
python train_fineweb.py \
    --model_dim 256 \
    --num_layers 4 \
    --num_experts 4 \
    --seq_len 128 \
    --batch_size 16 \
    --lr 3e-4 \
    --steps 1000
```

Dependencies:
  pip install torch datasets transformers tqdm
"""

import argparse
import itertools
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from fat_moe import MoETransformerLM  # assumes moe_llm.py is on PYTHONPATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stream_tokens(dataset_iter: Iterable[str], tokenizer, seq_len: int) -> Iterable[torch.Tensor]:
    """Yield contiguous sequences of *seq_len+1* tokens.

    The function packs raw text into a single long sequence of token IDs and
    then chunks it. No padding is needed and each chunk provides both input and
    target (next‑token) tokens.
    """
    buf: List[int] = []
    token_gen = (tokenizer(t["text"], add_special_tokens=False)["input_ids"] for t in dataset_iter)
    for toks in token_gen:
        buf.extend(toks + [tokenizer.eos_token_id])  # ensure separation
        while len(buf) > seq_len:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len + 1 :]
            yield torch.tensor(chunk, dtype=torch.long)


def batcher(token_stream: Iterable[torch.Tensor], batch_size: int) -> Iterable[torch.Tensor]:
    while True:
        batch = list(itertools.islice(token_stream, batch_size))
        if not batch:
            break
        yield torch.stack(batch)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer – we reuse GPT‑2 for convenience (Byte‑level BPE).
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # avoid size mismatch

    # Model
    model = MoETransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.model_dim,
        n_head=args.model_dim // 64,
        num_layers=args.num_layers,
        d_ff=args.ff_dim,
        num_experts=args.num_experts,
        max_seq_len=args.seq_len,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Streaming FineWeb‑Edu dataset
    print("Loading FineWeb‑Edu… (streaming mode)")
    dataset = load_dataset("google/wiki40b", "en", split="train", streaming=True)
    # dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    token_stream = stream_tokens(dataset, tokenizer, args.seq_len)
    batch_stream = batcher(token_stream, args.batch_size)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(args.steps), desc="Steps")
    for step in pbar:
        try:
            batch = next(batch_stream).to(device)  # (B, seq_len+1)
        except StopIteration:
            # Dataset exhausted – restart iterator for another pass.
            token_stream = stream_tokens(dataset, tokenizer, args.seq_len)
            batch_stream = batcher(token_stream, args.batch_size)
            batch = next(batch_stream).to(device)

        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits, aux = model(inputs)
        loss_main = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
        loss = loss_main + 0.01 * aux

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        pbar.set_postfix({"loss": loss_main.item(), "aux": aux.item()})

        if (step + 1) % 100 == 0 or step == args.steps - 1:
            ckpt_path = args.save_dir / f"step_{step+1}.pt"
            torch.save({
                "model": model.state_dict(),
                "opt": optim.state_dict(),
                "step": step + 1,
            }, ckpt_path)
            pbar.write(f"Checkpoint saved to {ckpt_path}")

    print("Training completed.")


if __name__ == "__main__":
    main()
