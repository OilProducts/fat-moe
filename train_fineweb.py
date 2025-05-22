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
import logging
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from fat_moe import MoETransformerLM  # assumes moe_llm.py is on PYTHONPATH

try:
    from aim import Run  # type: ignore
except ImportError:  # Degrade gracefully if Aim isn't available
    Run = None  # type: ignore[misc,assignment]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    fmt = "% (asctime)s | %(levelname)8s | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, stream=sys.stdout)
    logger = logging.getLogger("train")
    # Remove root handlers added by other libs (datasets transformers etc.) to avoid duplicates
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.propagate = False
    return logger

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
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--layer_repetition", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_attn_experts", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--eval_tokens", type=int, default=1024*8, help="Number of tokens from WikiText‑2 for perplexity")
    parser.add_argument("--sample_prompt", type=str, default="The purpose of education is")
    parser.add_argument("--sample_tokens", type=int, default=60)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log_every", type=int, default=10, help="Logging frequency (in steps)")


    args = parser.parse_args()
    logger = _setup_logger()

    aim_run: Optional[Run] = None
    if Run is not None:
        aim_run = Run(experiment="fineweb‑moe")
        aim_run["hparams"] = vars(args)
        logger.info("Aim tracking enabled ‑ run hash: %s", aim_run.hash)
    else:
        logger.warning("Aim is not installed; metrics will not be tracked.")

    torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # scaler = None
    # if device == "cuda":
    #     scaler = GradScaler()

    # Tokenizer ‑ GPT‑2 BPE
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = args.seq_len + 1  # disable GPT‑2 1024‑token warning
    if hasattr(tokenizer, "deprecation_warnings"):
        tokenizer.deprecation_warnings["sequence_length_is_longer_than_the_specified_max"] = None
    tokenizer.pad_token = tokenizer.eos_token  # just in case

    logger.info("Initializing with args: %s", vars(args))

    # Model
    model = MoETransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.model_dim,
        n_head=args.num_heads,
        num_layers=args.num_layers,
        layer_repetition=args.layer_repetition,
        d_ff=args.ff_dim,
        num_experts=args.num_experts,
        num_attn_experts=args.num_attn_experts,
        top_k=args.top_k,
        max_seq_len=args.seq_len,
    ).to(device)

    # model = torch.compile(model)

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_params = sum(p.numel() for p in model.layers.parameters() if p.requires_grad)
    effective_total = (num_params - layer_params) + layer_params * args.layer_repetition


    per_expert = sum(p.numel() for p in model.layers[0].moe_ff.experts[0].parameters())
    active_per_ff = per_expert * (args.num_experts - args.top_k)
    one_layer = sum(p.numel() for p in model.layers[0].parameters())
    active_param_layer = (one_layer - (per_expert*args.num_experts)) + per_expert * args.top_k
    without_layers = num_params - (one_layer * (args.num_layers + 2))  # 2 for initial and final
    one_tok = (without_layers + (active_param_layer * 2)) # + 2 for initial and final
    one_tok += (active_param_layer * args.num_layers) * args.layer_repetition


    logger.info("Model has %s parameters", short_num(num_params))
    logger.info("Model has %s effective parameters", short_num(effective_total))
    logger.info("Model has %s active parameters for ONE token (k=%d).", short_num(one_tok), args.top_k)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Streaming FineWeb‑Edu dataset
    logger.info("Loading FineWeb‑Edu... (this may take a moment on first run)")
    # dataset = load_dataset("google/wiki40b", "en", split="train", streaming=True)
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    token_stream = stream_tokens(dataset, tokenizer, args.seq_len)
    batch_stream = batcher(token_stream, args.batch_size)


    # Evaluation dataset (small slice of WikiText‑2)
    logger.info("Preparing WikiText‑2 slice for perplexity eval...")
    wikitext_iter = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    wt_tokens: List[torch.Tensor] = []
    for chunk in stream_tokens(wikitext_iter, tokenizer, args.seq_len):
        wt_tokens.append(chunk)
        if len(wt_tokens) * (args.seq_len + 1) >= args.eval_tokens:
            break


    args.save_dir.mkdir(parents=True, exist_ok=True)
    epoch_step_count = 0
    total_epochs_passed = 0

    # header = f"{'Step':>6s}/{args.steps:<6d} | Loss: {'Loss':>8s} | Aux: {'Aux':>8s} | Epoch: {'Epoch':>5s}"
    # logger.info(header)
    # logger.info("‑" * len(header))
    for step in range(args.steps):
        epoch_step_count += 1
        try:
            batch = next(batch_stream).to(device)  # (B, seq_len+1)
        except StopIteration:
            logger.info("Data iterator exhausted after %d steps; restarting stream.", epoch_step_count)
            epoch_step_count = 0
            total_epochs_passed += 1  # Optional
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

        if step % args.log_every == 0 or step == args.steps - 1:
            logger.info(
                f"Step: {step + 1:6d}/{args.steps:<6d} | Loss: {loss_main.item():8.4f} | Aux: {aux.item():8.4f} | Epoch: {total_epochs_passed:5d}"
            )
            stats = model.token_statistics()
            logger.info(
                "TOKENS total=%s\nattn=%s\n ffn=%s",
                f"{stats['total']:,}",
                ", ".join(f"E{i:02}:{short_num(c):9}" for i, c in enumerate(stats.get("attn_by_exp") or [])) or "‑",
                ", ".join(f"E{i:02}:{short_num(c):9}" for i, c in enumerate(stats["ffn_by_exp"]))
            )
            if aim_run is not None:
                aim_run.track(loss_main.item(), name="loss", step=step)
                aim_run.track(aux.item(), name="aux_loss", step=step)

        if step % 100 == 0:
            # Qualitative sample
            model.eval()
            sample_text = generate(
                model,
                tokenizer,
                prompt=args.sample_prompt,
                max_new_tokens=args.sample_tokens,
                temperature=0.2,
                top_k=50,
                device=device,
            )
            logger.info("\n" + "‑" * 80 + "\nSample:\n%s\n%s\n", sample_text, "‑" * 80)
            model.train()

            # Perplexity
            ppl = evaluate_perplexity(model, wt_tokens, device, tokenizer.vocab_size)
            logger.info("Perplexity on %d WikiText‑2 tokens: %.2f", len(wt_tokens) * (args.seq_len + 1), ppl)
            if aim_run is not None:
                aim_run.track(ppl, name="wikitext_perplexity", step=step)


        # --------------------------------------------------------------
        # Checkpoint + sampling
        # --------------------------------------------------------------
        if (step + 1) % 1000 == 0 or step == args.steps - 1:
            ckpt_path = args.save_dir / f"step_{step + 1}.pt"
            torch.save({"model": model.state_dict(), "opt": optim.state_dict(), "step": step + 1}, ckpt_path)
            logger.info("✅ Checkpoint saved to %s", ckpt_path)



    print("🎉 Training completed.")


if __name__ == "__main__":
    main()

