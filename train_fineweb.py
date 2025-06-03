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
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from fat_moe import MoETransformerLM, MoEFeedForward

try:
    from aim import Run  # type: ignore
except ImportError:  # Degrade gracefully if Aim isn't available
    Run = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    fmt = "%(asctime)s | %(levelname)8s | %(message)s"
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
            buf = buf[seq_len + 1:]
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


def generate(model: MoETransformerLM, tokenizer, prompt: str, max_new_tokens: int = 64, temperature: float = 1.0,
             top_k: int = 50, device="cpu") -> str:
    """Greedy/top‑k generator for quick qualitative checks."""
    tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokens = tokens[-model.max_seq_len:]

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


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))

    # Compute days, hours, minutes, and seconds.
    days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, secs = divmod(remainder, 60)  # 60 seconds in a minute

    parts = []
    # If there are days, show days, hours, and minutes (ignore seconds)
    if days > 0:
        parts.append(f"{days}d")
        parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
    # If there are hours but no days, show hours and minutes (ignore seconds)
    elif hours > 0:
        parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
    # If it's less than one hour, show minutes and seconds (or only seconds if under a minute)
    else:
        if minutes > 0:
            parts.append(f"{minutes}m")
            parts.append(f"{secs}s")
        else:
            parts.append(f"{secs}s")

    return " ".join(parts)


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


def noise_schedule(step: int, total: int,
                   warmup: float = 0.10, decay: float = 0.30,
                   start: float = 1.0, end: float = 0.0) -> float:
    if step < warmup * total:
        return start
    t = (step - warmup * total) / (decay * total)
    if t < 1.0:
        return start * (1 - t) + end * t  # linear
    return end


def set_noise_std(model, value: float) -> None:
    """Recursively set `noise_std` on every MoE FFN in `model`."""
    for mod in model.modules():
        if isinstance(mod, MoEFeedForward):
            mod.noise_std = value


def pack_sequences(examples,
                   chunk_len: int = 513,  # seq_len + 1
                   eos_id: int | None = None):
    """
    Turn a batch of variable‑length `input_ids` into fixed‑length blocks.

    Returns
    -------
    dict: {"input_ids": List[List[int]]}
          (the outer list = new rows, the inner list = token ids)
    """
    # 1. flatten + add <EOS>
    stream: list[int] = []
    for ids in examples["input_ids"]:
        stream.extend(ids)
        if eos_id is not None:
            stream.append(eos_id)

    # 2. drop the tail that doesn’t fill a full block (optional)
    total = len(stream) // chunk_len * chunk_len
    stream = stream[:total]

    # 3. split into blocks
    blocks = [stream[i: i + chunk_len]
              for i in range(0, total, chunk_len)]

    # 4. HuggingFace will expand the outer list into new rows
    return {"input_ids": blocks}

def pack_and_tokenise(batch,
                      tokenizer,
                      chunk_len=513,          # global fast‑tokeniser
                      eos_id=None,
                      ):
    # 1) tokenise all texts in the batch at once
    ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]

    # 2) flatten & add <eos>
    stream = list(itertools.chain.from_iterable(ids))
    if eos_id is not None:
        stream = list(itertools.chain.from_iterable(
            (seq + [eos_id] for seq in ids)
        ))

    # 3) chop to fixed blocks
    total = len(stream) // chunk_len * chunk_len
    blocks = [stream[i:i+chunk_len] for i in range(0, total, chunk_len)]
    return {"input_ids": blocks}



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--layer_repetition", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_attn_experts", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval_tokens", type=int, default=1024 * 64,
                        help="Number of tokens from WikiText‑2 for perplexity")
    parser.add_argument("--sample_prompt", type=str, default="The purpose of education is")
    parser.add_argument("--sample_tokens", type=int, default=60)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log_every", type=int, default=10, help="Logging frequency (in steps)")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Micro‑batches to accumulate before an optimizer step")
    parser.add_argument("--capacity_factor", type=float, default=1.25)

    args = parser.parse_args()
    logger = _setup_logger()

    aim_run: Optional[Run] = None
    if Run is not None:
        aim_run = Run(experiment="fineweb-moe")
        for k, v in vars(args).items():
            if k not in ["save_dir", "sample_prompt", "sample_tokens"]:
                aim_run[k] = v
        # aim_run["hparams"] = args
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
    active_param_layer = (one_layer - (per_expert * args.num_experts)) + per_expert * args.top_k
    without_layers = num_params - (one_layer * (args.num_layers + 2))  # 2 for initial and final
    one_tok = (without_layers + (active_param_layer * 2))  # + 2 for initial and final
    one_tok += (active_param_layer * args.num_layers) * args.layer_repetition

    logger.info("Model has %s parameters", short_num(num_params))
    logger.info("Model has %s effective parameters", short_num(effective_total))
    logger.info("Model has %s active parameters for ONE token (k=%d).", short_num(one_tok), args.top_k)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Streaming FineWeb‑Edu dataset
    logger.info("Loading FineWeb‑Edu... (this may take a moment on first run)")
    # dataset = load_dataset("google/wiki40b", "en", split="train", streaming=True)
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

    tokenised = (dataset
        .map(
            pack_and_tokenise,
            batched=True,
            batch_size=1_000,
            num_proc = int(os.cpu_count()/2),
            fn_kwargs={"tokenizer": tokenizer, "chunk_len": args.seq_len + 1, "eos_id": tokenizer.eos_token_id},
            remove_columns=dataset.column_names)).with_format("torch")

    chunk_len = args.seq_len + 1
    # packed = tokenised.map(pack_sequences,
    #                        batched=True,
    #                        batch_size=1000,
    #                        remove_columns=tokenised.column_names,
    #                        fn_kwargs={"chunk_len": chunk_len, "eos_id": tokenizer.eos_token_id}).with_format("torch")

    loader = DataLoader(tokenised,
                        batch_size=args.batch_size,
                        num_workers=int(os.cpu_count()/2),
                        pin_memory=True,
                        drop_last=True,
                        collate_fn=lambda batch: torch.stack(
                            [row["input_ids"] for row in batch]))  # GPU‑friendly batches
    def infinite_loader(loader: DataLoader) -> Iterable[torch.Tensor]:
        """Yield batches endlessly."""
        while True:
            for batch in loader:
                yield batch
    batch_iter = infinite_loader(loader)  # endless stream

    # Evaluation dataset (small slice of WikiText‑2)
    logger.info("Preparing WikiText‑2 slice for perplexity eval...")
    wikitext_iter = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wt_tokens: List[torch.Tensor] = []
    for chunk in stream_tokens(wikitext_iter, tokenizer, args.seq_len):
        wt_tokens.append(chunk)
        if len(wt_tokens) * (args.seq_len + 1) >= args.eval_tokens:
            break

    args.save_dir.mkdir(parents=True, exist_ok=True)

    tok_seen = 0
    start_time = time.time()
    last_log_time = start_time
    last_tok = 0

    global_step = 0  # counts **optimizer** steps
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=2000, num_training_steps=args.steps,
                                                num_cycles=0.5)

    while global_step < args.steps:
        current_std = noise_schedule(global_step, args.steps)
        set_noise_std(model, current_std)
        total_loss = 0.0

        for micro in range(args.grad_accum):
            batch = next(batch_iter).to(device, non_blocking=True)  # (B, seq_len+1)

            inputs, targets = batch[:, :-1], batch[:, 1:]
            B, S = inputs.shape
            logits, aux = model(inputs)
            loss_main = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                targets.reshape(-1),
            )
            micro_loss = (loss_main + .1 * aux) / args.grad_accum
            tok_seen += B * S
            micro_loss.backward()
            total_loss += micro_loss

        optim.step()
        scheduler.step()
        optim.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % args.log_every == 0 or global_step == args.steps - 1:
            now = time.time()
            interval_toks = tok_seen - last_tok
            interval_time = now - last_log_time
            tps = interval_toks / interval_time
            last_log_time = now
            last_tok = tok_seen
            steps_left = args.steps - global_step
            eta_seconds = (steps_left / args.log_every) * interval_time

            # Main training stats
            logger.info(
                f"Step: {global_step + 1:6d}/{args.steps:<6d} | "
                f"Loss: {loss_main.item():8.4f} | "
                f"Aux: {aux.item():8.4f} | "  # This 'aux' comes from MoEFeedForward.forward()
                f"tok/s: {short_num(int(tps))} | "
                f"ETA: {format_duration(eta_seconds)} | "
            )

            stats = model.token_statistics()  # This now includes *_gate_selection_by_exp

            # --- Aggregate Token Statistics ---
            logger.info(
                "TOKEN STATS (Aggregate): total_model_input=%s",
                short_num(stats['total'])
            )

            # Aggregate FFN Stats
            agg_ffn_routed_str = "N/A"
            if stats.get("ffn_by_exp") is not None:
                agg_ffn_routed_str = ", ".join(
                    f"E{i:02}:{short_num(c):>6}" for i, c in enumerate(stats["ffn_by_exp"]))

            agg_ffn_selected_str = "N/A"
            if stats.get("ffn_gate_selection_by_exp") is not None:
                agg_ffn_selected_str = ", ".join(f"E{i:02}:{short_num(c):>6}" for i, c in
                                                 enumerate(stats["ffn_gate_selection_by_exp"]))

            logger.info(
                f"  FFN Agg: Routed=[{agg_ffn_routed_str}] | SelectedByGate=[{agg_ffn_selected_str}]")

            # Aggregate ATTN Stats (Optional, if you have MoE Attention and want to log it)
            if stats.get("attn_by_exp") is not None or stats.get(
                    "attn_gate_selection_by_exp") is not None:
                agg_attn_routed_str = "N/A"
                if stats.get("attn_by_exp") is not None:
                    agg_attn_routed_str = ", ".join(
                        f"E{i:02}:{short_num(c):>6}" for i, c in enumerate(stats["attn_by_exp"]))

                agg_attn_selected_str = "N/A"
                if stats.get("attn_gate_selection_by_exp") is not None:
                    agg_attn_selected_str = ", ".join(f"E{i:02}:{short_num(c):>6}" for i, c in
                                                      enumerate(
                                                          stats["attn_gate_selection_by_exp"]))
                logger.info(
                    f"  ATTN Agg: Routed=[{agg_attn_routed_str}] | SelectedByGate=[{agg_attn_selected_str}]")

            # --- Per-Layer Expert Statistics ---
            logger.info("-" * 80)  # Separator
            logger.info(
                "Per-Layer Expert Statistics (Name | Type | Total Layer In | Routed | Selected by Gate):")
            for layer_stat in stats[
                "layers"]:  # Renamed 'layer' to 'layer_stat' to avoid conflict if 'layer' is a variable
                name_str = layer_stat["name"].ljust(7)

                # FFN Stats for the layer
                ffn_total_layer = layer_stat.get("ffn_total", 0)
                ffn_routed_list = layer_stat.get("ffn_by_exp")
                ffn_selected_list = layer_stat.get("ffn_gate_selection_by_exp")

                if ffn_routed_list is not None or ffn_selected_list is not None:  # Only print if FFN MoE stats exist
                    ffn_total_str = short_num(ffn_total_layer)
                    ffn_routed_str = "N/A"
                    if ffn_routed_list is not None:
                        ffn_routed_str = "[" + ", ".join(
                            f"E{i}:{short_num(c):>5}" for i, c in enumerate(ffn_routed_list)) + "]"

                    ffn_selected_str = "N/A"
                    if ffn_selected_list is not None:
                        ffn_selected_str = "[" + ", ".join(f"E{i}:{short_num(c):>5}" for i, c in
                                                           enumerate(ffn_selected_list)) + "]"
                    logger.info(
                        f"{name_str} | FFN    | In:{ffn_total_str:>6} | {ffn_routed_str} | {ffn_selected_str}")

                # ATTN Stats for the layer (Optional)
                attn_total_layer = layer_stat.get("attn_total", 0)
                attn_routed_list = layer_stat.get("attn_by_exp")
                attn_selected_list = layer_stat.get("attn_gate_selection_by_exp")

                if attn_routed_list is not None or attn_selected_list is not None:  # Only print if ATTN MoE stats exist
                    attn_total_str = short_num(attn_total_layer)
                    attn_routed_str = "N/A"
                    if attn_routed_list is not None:
                        attn_routed_str = "[" + ", ".join(
                            f"E{i}:{short_num(c):>5}" for i, c in enumerate(attn_routed_list)) + "]"

                    attn_selected_str = "N/A"
                    if attn_selected_list is not None:
                        attn_selected_str = "[" + ", ".join(f"E{i}:{short_num(c):>5}" for i, c in
                                                            enumerate(attn_selected_list)) + "]"
                    logger.info(
                        f"{name_str} | ATTN   | In:{attn_total_str:>6} | {attn_routed_str} | {attn_selected_str}")
            logger.info("-" * 80)  # Separator

            if aim_run is not None:
                aim_run.track(loss_main.item(), name="loss", step=global_step)
                aim_run.track(aux.item(), name="aux_loss", step=global_step)
                # You could also track aggregated expert stats in Aim if desired:
                # if stats.get("ffn_by_exp") is not None:
                #     for i, count in enumerate(stats["ffn_by_exp"]):
                #         aim_run.track(count, name=f"agg_ffn_routed_exp{i}", step=global_step, context={"type": "expert_routing"})
                # if stats.get("ffn_gate_selection_by_exp") is not None:
                #     for i, count in enumerate(stats["ffn_gate_selection_by_exp"]):
                #         aim_run.track(count, name=f"agg_ffn_selected_exp{i}", step=global_step, context={"type": "expert_routing"})

        if global_step % 100 == 0:
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
                aim_run.track(ppl, name="wikitext_perplexity", step=global_step)

        # --------------------------------------------------------------
        # Checkpoint + sampling
        # --------------------------------------------------------------
        if (global_step + 1) % 1000 == 0 or global_step == args.steps - 1:
            ckpt_path = args.save_dir / f"step_{global_step + 1}.pt"
            torch.save({"model": model.state_dict(), "opt": optim.state_dict(), "step": global_step + 1}, ckpt_path)
            logger.info("✅ Checkpoint saved to %s", ckpt_path)

    print("🎉 Training completed.")


if __name__ == "__main__":
    main()
