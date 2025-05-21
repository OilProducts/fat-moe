import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



class MoEFeedForward(nn.Module):
    """Minimal Mixture‑of‑Experts feed‑forward module with top-k routing.

    Each token is routed to and processed by `k` selected experts.
    The outputs of these `k` experts are combined using learned soft gating
    weights (probabilities from a softmax over the top-k expert scores).
    This implementation performs sparse computation, i.e., only selected
    experts compute for a given token. It includes a load balancing
    auxiliary loss to encourage expert specialization.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model # Store d_model for reshaping output
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
        # Probabilities based on EMA counts (how often each expert was selected)
        probs = self._ema_counts / (self._ema_counts.sum() + 1e-9)
        # Loss encourages probabilities to be uniform (1/num_experts)
        # Higher value means more imbalance. Ranges from 1 (balanced) to num_experts (imbalanced).
        return (probs * probs).sum() * self.num_experts

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor (batch_size, seq_len, d_model)
            k: number of experts to route each token to
        Returns:
            y: output tensor (batch_size, seq_len, d_model)
            aux_loss: load balancing loss
        """
        batch_size, seq_len, _ = x.shape
        num_tokens = batch_size * seq_len

        # 1. Gating: Get scores and select top-k experts
        gate_logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        topk_expert_scores, topk_expert_indices = gate_logits.topk(k, dim=-1)  # (batch_size, seq_len, k) for both

        # 2. EMA Update for Load Balancing (based on hard assignments)
        # Create a mask indicating which experts were selected for any token
        # This mask (one-hot representation of topk_expert_indices) is used for load balancing stats
        mask_for_ema = torch.zeros_like(gate_logits, dtype=torch.float32).scatter_(
            -1, topk_expert_indices, 1.0
        ) # (batch_size, seq_len, num_experts)
        with torch.no_grad():
            # Calculate mean usage for this batch (fraction of times each expert was selected)
            # batch_mean_usage will have shape (num_experts,)
            # The sum of batch_mean_usage will be k, as each token selects k experts.
            batch_mean_usage = mask_for_ema.mean(dim=(0, 1))
            self._ema_counts.mul_(self.ema_decay).add_(batch_mean_usage, alpha=1 - self.ema_decay)

        # 3. Get Gating Probabilities (weights for combining expert outputs)
        # Softmax over the scores of the selected top-k experts.
        # These probabilities are used to weight the outputs of the chosen experts.
        # Gradients will flow through these probabilities to train the gate.
        gating_probabilities = F.softmax(topk_expert_scores, dim=-1)  # (batch_size, seq_len, k)

        # 4. Sparse Dispatch and Expert Computation
        # Flatten inputs and routing information for easier processing
        flat_x = x.reshape(num_tokens, self.d_model)                        # (num_tokens, d_model)
        flat_topk_expert_indices = topk_expert_indices.reshape(num_tokens, k) # (num_tokens, k)
        flat_gating_probabilities = gating_probabilities.reshape(num_tokens, k) # (num_tokens, k)

        # Create dispatch tensors:
        # - `dispatch_expert_indices`: For each of (num_tokens*k) choices, which expert it is (0 to num_experts-1).
        # - `dispatch_token_indices`: For each of (num_tokens*k) choices, which token it belongs to (0 to num_tokens-1).
        # - `dispatch_weights`: For each of (num_tokens*k) choices, its gating probability.
        dispatch_expert_indices = flat_topk_expert_indices.flatten() # Shape: (num_tokens * k)
        dispatch_token_indices = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(num_tokens, k)
            .flatten()
        ) # Shape: (num_tokens * k)
        dispatch_weights = flat_gating_probabilities.flatten() # Shape: (num_tokens * k)

        # Initialize the output tensor for flattened tokens
        y_flat = torch.zeros_like(flat_x) # (num_tokens, d_model)

        # Iterate over each expert, process its assigned tokens, and accumulate weighted outputs
        for expert_idx in range(self.num_experts):
            # Find all routing choices that dispatch to the current expert
            expert_active_mask = (dispatch_expert_indices == expert_idx)

            if expert_active_mask.any(): # If this expert is used for any token choice
                # Get the original token indices (in flat_x) that need to be processed by this expert
                tokens_for_this_expert_indices = dispatch_token_indices[expert_active_mask]

                # Get the actual token data for this expert
                inputs_for_this_expert = flat_x[tokens_for_this_expert_indices] # (num_routed_to_expert, d_model)

                # Get the gating weights for these tokens for this expert
                weights_for_this_expert = dispatch_weights[expert_active_mask].unsqueeze(1) # (num_routed_to_expert, 1)

                # Compute expert output
                expert_output = self.experts[expert_idx](inputs_for_this_expert) # (num_routed_to_expert, d_model)

                # Apply weights and add to the corresponding token's output sum using index_add_
                # index_add_ correctly sums contributions if a token has multiple slots routed to the same expert (though not typical with topk)
                y_flat.index_add_(0, tokens_for_this_expert_indices, expert_output * weights_for_this_expert)

        # 5. Reshape output to original batch and sequence dimensions
        y = y_flat.reshape(batch_size, seq_len, self.d_model)

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
        # cache causal mask (non‑persistent so it adjusts to device)
        self.register_buffer("_causal_mask", torch.empty(0), persistent=False)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask.shape[:2] != (seq_len, seq_len):
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
            self._causal_mask = mask
        return self._causal_mask

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self‑attention with causal mask
        mask = self._get_causal_mask(x.size(1), x.device)
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
