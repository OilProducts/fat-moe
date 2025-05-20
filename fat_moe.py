import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFeedForward(nn.Module):
    """Minimal Mixture‑of‑Experts feed‑forward module.

    Each token is processed by *all* experts and the outputs are combined
    with a soft gating weight. This keeps the implementation simple and
    fully differentiable while still letting you experiment with expert
    specialization and gating strategies.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        # Gating network maps each token to a distribution over experts.
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        # A tiny 2‑layer feed‑forward network per expert.
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff, bias=False),
                    nn.ReLU(),
                    nn.Linear(d_ff, d_model, bias=False),
                )
                for _ in range(num_experts)
            ]
        )
        # Exponential‑moving average of expert usage for load‑balancing.
        self.register_buffer("_ema_counts", torch.zeros(num_experts), persistent=False)
        self.ema_decay = 0.99

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def load_balance_loss(self) -> torch.Tensor:
        """Return a scalar auxiliary loss that penalises expert imbalance."""
        probs = self._ema_counts / (self._ema_counts.sum() + 1e-9)
        return (probs * probs).sum() * self.num_experts  # higher when imbalanced

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x, k: int = 2):
        gate_logits = self.gate(x)  # (b, s, E)
        topk_val, topk_idx = gate_logits.topk(k, dim=-1)  # (b, s, k)

        # Hard routing mask: 1 for chosen experts, 0 elsewhere
        mask = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, 1.0)

        # Softmax only over the chosen experts for normalisation
        gate_probs = F.softmax(topk_val, dim=-1)  # (b, s, k)
        gate_probs_full = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, gate_probs)

        # Straight‑through estimator:
        gate_probs_st = gate_probs_full + (mask - gate_probs_full).detach()

        # Track usage stats
        with torch.no_grad():
            self._ema_counts.mul_(self.ema_decay).add_(gate_probs_st.mean((0, 1)),
                                                       alpha=1 - self.ema_decay)

        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=0)  # (E, b, s, d)
        y = torch.einsum("bsE,Ebsd->bsd", gate_probs_st, expert_outs)

        return y, self.load_balance_loss()


class TransformerBlock(nn.Module):
    """A single Transformer block with an MoE feed‑forward layer."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe_ff = MoEFeedForward(d_model, d_ff, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self‑attention.
        sa_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(sa_out)
        x = self.ln1(x)

        # MoE feed‑forward.
        ff_out, aux = self.moe_ff(x)
        x = x + self.dropout(ff_out)
        x = self.ln2(x)
        return x, aux


class MoETransformerLM(nn.Module):
    """A minimal GPT‑style language model with Mixture‑of‑Experts FFNs."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_head: int = 4,
        num_layers: int = 2,
        d_ff: int = 1024,
        num_experts: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_head, d_ff, num_experts, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            idx: (batch, seq) token indices
        Returns:
            logits: (batch, seq, vocab)
            aux_loss: combined auxiliary MoE load‑balancing loss
        """
        B, S = idx.shape
        assert S <= self.max_seq_len, "Sequence too long"
        pos = torch.arange(0, S, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        aux_losses = 0.0
        for layer in self.layers:
            x, aux = layer(x)
            aux_losses = aux_losses + aux

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, aux_losses


# -----------------------------------------------------------------------------
# Tiny usage example / smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = 100
    model = MoETransformerLM(vocab, num_layers=2, num_experts=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Dummy data: predict next token in a random sequence.
    B, S = 8, 32
    data = torch.randint(0, vocab, (B, S + 1), device=device)
    inputs, targets = data[:, :-1], data[:, 1:]

    logits, aux = model(inputs)
    loss_main = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
    loss = loss_main + 0.01 * aux

    loss.backward()
    optimizer.step()

    print(
        f"Training step OK \u2013 total loss {loss.item():.4f} (main {loss_main.item():.4f}, aux {aux.item():.4f})"
    )
