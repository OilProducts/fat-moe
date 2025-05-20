"""Tiny training harness for MoETransformerLM on the streaming FineWeb‑EDU dataset.

Features
--------
* Streams dataset (no huge download)
* Saves checkpoints every *N* steps
* **New:** generates a sample continuation from a fixed prompt each time it
  checkpoints, so you can watch the model's progress in real‑time.

Example
~~~~~~~
```bash
python train_fineweb.py \
  --model_dim 256 --num_layers 4 --num_experts 4 \
  --seq_len 128 --batch_size 16 --steps 1000
```

Dependencies
~~~~~~~~~~~~
``pip install torch datasets transformers tqdm``
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from fat_moe import MoETransformerLM  # assumes moe_llm.py is on PYTHONPATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stream_tokens(dataset_iter: Iterable[dict], tokenizer, seq_len: int) -> Iterable[torch.Tensor]:
    """Pack incoming texts into contiguous `seq_len+1` chunks of token IDs."""
    buf: List[int] = []
    for sample in dataset_iter:
        buf.extend(tokenizer(sample["text"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id])
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


def top_k_sample(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> int:
    """Return an index sampled from the *top‑k* logits (1‑D tensor)."""
    logits = logits / temperature
    values, indices = torch.topk(logits, k)
    probs = torch.softmax(values, dim=-1)
    next_idx = torch.multinomial(probs, 1).item()
    return indices[next_idx].item()


def generate(model: MoETransformerLM, tokenizer, prompt: str, max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 50, device="cpu") -> str:
    """Greedy/top‑k generator for quick qualitative checks."""
    model.eval()
    tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokens = tokens[-model.max_seq_len :]

    for _ in range(max_new_tokens):
        idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(idx)
        next_token_logits = logits[0, -1]
        next_id = top_k_sample(next_token_logits, k=top_k, temperature=temperature)
        tokens.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(tokens)


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


def evaluate_perplexity(model: MoETransformerLM, token_chunks: List[torch.Tensor], device, vocab_size) -> float:
    """Compute perplexity on pre‑chunked token blocks.

    Args:
        token_chunks: list where each element is shape (seq_len+1,)
    Returns perplexity (float).
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for chunk in token_chunks:
            chunk = chunk.to(device)
            inputs, targets = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += targets.numel()
    return math.exp(total_loss / total_tokens)


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
    parser.add_argument("--sample_prompt", type=str, default="The purpose of education is")
    parser.add_argument("--sample_tokens", type=int, default=60)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer ‑ GPT‑2 BPE
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = args.seq_len + 1  # disable GPT‑2 1024‑token warning
    if hasattr(tokenizer, "deprecation_warnings"):
        tokenizer.deprecation_warnings["sequence_length_is_longer_than_the_specified_max"] = None
    tokenizer.pad_token = tokenizer.eos_token  # just in case

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

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {short_num(num_params)} parameters")

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
            token_stream = stream_tokens(dataset, tokenizer, args.seq_len)
            batch_stream = batcher(token_stream, args.batch_size)
            batch = next(batch_stream).to(device)

        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits, aux = model(inputs)
        loss_main = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            targets.reshape(-1),
        )
        loss = loss_main + 0.01 * aux

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        pbar.set_postfix({"loss": f"{loss_main.item():.4f}", "aux": f"{aux.item():.4f}"})

        # --------------------------------------------------------------
        # Checkpoint + sampling
        # --------------------------------------------------------------
        if (step + 1) % 100 == 0 or step == args.steps - 1:
            ckpt_path = args.save_dir / f"step_{step+1}.pt"
            torch.save({
                "model": model.state_dict(),
                "opt": optim.state_dict(),
                "step": step + 1,
            }, ckpt_path)
            pbar.write(f"✅ Checkpoint saved to {ckpt_path}")

            # Quick qualitative sample
            sample_text = generate(
                model,
                tokenizer,
                prompt=args.sample_prompt,
                max_new_tokens=args.sample_tokens,
                temperature=1.0,
                top_k=50,
                device=device,
            )
            pbar.write("\n" + "-" * 80)
            pbar.write("Sample:\n" + sample_text)
            pbar.write("-" * 80 + "\n")

    print("🎉 Training completed.")


if __name__ == "__main__":
    main()

