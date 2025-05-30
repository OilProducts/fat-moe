import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepeatEmbedding(nn.Module):
    def __init__(self, n_repeats: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_repeats, d_model)

    def forward(self, rep_idx: int, x: torch.Tensor):
        return x + self.emb.weight[rep_idx]


class MoEFeedForward(nn.Module):
    """Minimal Mixture‑of‑Experts feed‑forward module with top-k routing.
    (Docstring as before)
    """

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 num_experts: int,
                 noise_std: float = 1.0,
                 capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(d_model, num_experts, bias=False)
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

        # --- Cumulative Statistic Buffers ---
        # 1. Total tokens that entered the layer (B*S)
        self.register_buffer("_total_tokens",
                             torch.tensor(0, dtype=torch.long),
                             persistent=False)

        # 2. Cumulative tokens ACTUALLY ROUTED to each expert (post-capacity)
        # This will be used by your expert_token_stats for 'ffn_by_exp'
        self.register_buffer("_token_counts",  # Renaming for clarity based on usage
                             torch.zeros(num_experts, dtype=torch.float32),
                             persistent=False)

        # 3. Cumulative tokens SELECTED BY THE GATE for each expert (pre-capacity)
        # Useful for observing gate's raw preferences and calculating "diff"
        self.register_buffer("_gate_selection_counts",
                             torch.zeros(num_experts, dtype=torch.float32),
                             persistent=False)

        # --- EMA Statistic Buffer (Post-Capacity) ---
        # EMA of sum of gating probabilities for tokens dispatched to each expert.
        self.register_buffer("_ema_importance",
                             torch.zeros(num_experts, dtype=torch.float32),
                             persistent=False)

        self.ema_decay = 0.99

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def load_balance_loss(self) -> torch.Tensor:
        """
        Return a scalar auxiliary loss penalising expert imbalance based on
        cumulative historical counts of ACTUALLY ROUTED tokens (post-capacity).
        """
        # self._token_counts now refers to actual dispatch counts (post-capacity)
        if self._token_counts.sum() == 0:  # Avoid division by zero
            return torch.tensor(0.0, device=self._token_counts.device)

        # probs_actual_dispatch: historical probability of an expert actually processing a token
        probs_actual_dispatch = self._token_counts / (self._token_counts.sum() + 1e-9)
        return (probs_actual_dispatch * probs_actual_dispatch).sum() * self.num_experts

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        E = self.num_experts
        num_tokens_total_batch = B * S

        # --- Update total input tokens ---
        with torch.no_grad():
            self._total_tokens += num_tokens_total_batch

        # ------------------------------------------------ 1. Gate logits & Top-k Selection
        gate_logits = self.gate(x)
        if self.training and self.noise_std > 0.0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std

        topk_scores, topk_idx = gate_logits.topk(k, dim=-1)  # topk_idx shape: (B,S,k)
        probs = torch.softmax(topk_scores, dim=-1)

        # ------------------------------------------------ 1.5 Update Gate Selection Counts (Pre-Capacity)
        with torch.no_grad():
            for slot_idx in range(k):
                choices_for_slot = topk_idx[:, :, slot_idx].flatten()
                expert_ids_in_slot, freq_in_slot = choices_for_slot.unique(return_counts=True)
                self._gate_selection_counts.index_add_(
                    0, expert_ids_in_slot, freq_in_slot.to(self._gate_selection_counts.dtype)
                )

        # ------------------------------------------------ 2. Soft‑capacity mask
        # (Logic for capacity, keep_mask, and re-normalizing probs remains the same)
        capacity = math.ceil(self.capacity_factor * num_tokens_total_batch / E)
        flat_topk_idx = topk_idx.reshape(-1)
        one_hot_dispatch = F.one_hot(flat_topk_idx, num_classes=E).float()
        tokens_per_expert_cumulative = one_hot_dispatch.cumsum(dim=0)
        capacity_check = tokens_per_expert_cumulative <= capacity
        flat_keep_mask_for_slot = capacity_check.gather(1, flat_topk_idx.unsqueeze(1)).squeeze(1)
        keep_mask = flat_keep_mask_for_slot.view(B, S, k)

        probs = probs * keep_mask
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs / (probs_sum + 1e-9)

        # ------------------------------------------------ 3. Dispatch tensors
        # (Logic for dispatch_tokens_global_idx, dispatch_experts_idx, dispatch_weights remains the same)
        flat_probs = probs.reshape(-1)
        nonzero_dispatch_mask = flat_probs > 0
        original_token_indices = torch.arange(num_tokens_total_batch, device=x.device)
        dispatch_tokens_global_idx = original_token_indices.repeat_interleave(k)[
            nonzero_dispatch_mask]
        dispatch_experts_idx = flat_topk_idx[
            nonzero_dispatch_mask]  # Expert indices for dispatched tokens
        dispatch_weights = flat_probs[nonzero_dispatch_mask]

        x_flat = x.reshape(num_tokens_total_batch, self.d_model)
        x_dispatch_input = x_flat[dispatch_tokens_global_idx]

        # ------------------------------------------------ 4. Expert execution
        # (Logic remains the same)
        y_flat = torch.zeros_like(x_flat)
        for e_idx in range(E):
            expert_mask = (dispatch_experts_idx == e_idx)
            if expert_mask.any():
                tokens_for_expert_e = x_dispatch_input[expert_mask]
                weights_for_expert_e = dispatch_weights[expert_mask].unsqueeze(1)
                expert_output_e = self.experts[e_idx](tokens_for_expert_e)
                global_indices_for_expert_e = dispatch_tokens_global_idx[expert_mask]
                y_flat.index_add_(0, global_indices_for_expert_e,
                                  weights_for_expert_e * expert_output_e)
        y = y_flat.view(B, S, self.d_model)

        # ------------------------------------------------ 5. Batch/EMA Statistics (Post-Capacity) + Switch Aux Loss
        # Count of tokens ACTUALLY ROUTED to each expert in this batch
        batch_tokens_actually_routed_per_expert = torch.bincount(
            dispatch_experts_idx, minlength=E
        ).float()

        # Sum of gating probabilities for tokens ACTUALLY ROUTED in this batch
        batch_importance_sum_per_expert = torch.zeros(E, device=x.device, dtype=x.dtype)
        batch_importance_sum_per_expert.index_add_(0, dispatch_experts_idx, dispatch_weights)

        # --- Update Cumulative Actual Dispatch Counts (Post-Capacity) ---
        with torch.no_grad():
            # self._token_counts is for 'ffn_by_exp' -> actual routed tokens
            self._token_counts += batch_tokens_actually_routed_per_expert

        # --- Update EMA for importance (post-capacity) ---
        self._ema_importance.mul_(self.ema_decay).add_(
            batch_importance_sum_per_expert, alpha=1 - self.ema_decay
        )

        # --- Switch-Transformer load‑balancing loss (for training) ---
        # Based on current batch post-capacity stats
        tokens_frac_batch = batch_tokens_actually_routed_per_expert / \
                            (batch_tokens_actually_routed_per_expert.sum() + 1e-9)
        importance_frac_batch = batch_importance_sum_per_expert / \
                                (batch_importance_sum_per_expert.sum() + 1e-9)
        aux_loss_switch = (tokens_frac_batch * importance_frac_batch).sum() * E

        # --- Choose your auxiliary loss for training ---
        final_aux_loss = aux_loss_switch  # Default to Switch loss
        # Example: use historical actual dispatch imbalance
        # final_aux_loss = self.load_balance_loss()

        return y, final_aux_loss


class MoESelfAttention(nn.Module):
    """
    Multi‑Head Self‑Attention where a small gate chooses the top‑k attention
    *experts* (each expert is a standard nn.MultiheadAttention).
    The whole sequence is sent to the selected experts and their
    outputs are combined with soft weights from the gate.
    """

    def __init__(
            self,
            d_model: int,
            n_head: int,
            num_experts: int,
            dropout: float = 0.1,
            k: int = 2,
            ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_experts, self.k = num_experts, k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.attn_experts = nn.ModuleList(
            nn.MultiheadAttention(
                d_model, n_head, dropout=dropout, batch_first=True
            )
            for _ in range(num_experts)
        )
        self.register_buffer("_token_counts", torch.zeros(num_experts, dtype=torch.long),
                             persistent=False)  # tokens routed to every expert
        self.register_buffer("_total_tokens", torch.tensor(0, dtype=torch.long),
                             persistent=False)  # all tokens that passed the module
        self.register_buffer("_ema_counts", torch.zeros(num_experts), persistent=False)
        self.ema_decay = ema_decay
        self.dropout = nn.Dropout(dropout)

    # --------- helper identical to FFN load‑balance -----------------
    def load_balance_loss(self) -> torch.Tensor:
        probs = self._ema_counts / (self._ema_counts.sum() + 1e-9)
        return (probs * probs).sum() * self.num_experts

    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, S, D)
        causal_mask: (S,S) – pre‑computed float(‑inf) mask you already cache

        Returns
        -------
        y          : (B,S,D)  – weighted mixture of expert outputs
        aux_lb_loss: scalar      – expert‑load‑balancing term
        """
        B, S, D = x.shape

        # --- 1. gating ------------------------------------------------
        gate_logits = self.gate(x)  # (B,S,E)
        # Pool over sequence so one routing decision per *example*
        pooled = gate_logits.mean(dim=1)  # (B,E)
        scores, indices = pooled.topk(self.k, dim=-1)  # (B,k)

        # Update cumulative counters
        with torch.no_grad():
            B, S = x.shape[:2]
            self._total_tokens += B * S

            # each example’s whole sequence is sent to `indices[b, slot]`
            for slot in range(self.k):
                exp_ids, exp_freq = indices[:, slot].unique(return_counts=True)
                # add (#examples · S) for every chosen expert
                self._token_counts.index_add_(0, exp_ids,
                                              exp_freq.to(self._token_counts.dtype) * S)

        # record EMA usage (hard assignment)
        mask_for_ema = F.one_hot(indices.reshape(-1), self.num_experts) \
            .float().sum(0)
        with torch.no_grad():
            self._ema_counts.mul_(self.ema_decay).add_(mask_for_ema / B, alpha=1 - self.ema_decay)

        probs = F.softmax(scores, dim=-1)  # (B,k)

        # --- 2. run selected experts ---------------------------------
        expert_outs = []
        # because k is tiny (1–2) this loop is fine
        for slot in range(self.k):
            expert_idx = indices[:, slot]  # (B,)
            # find unique experts in this slot to avoid redundant calls
            unique = expert_idx.unique()
            slot_out = torch.zeros_like(x)
            for e in unique:
                e_mask = expert_idx == e  # which batch items choose this expert
                if e_mask.any():
                    x_e = x[e_mask]  # gather
                    # MultiheadAttention expects (N, S, D) queries/keys/values
                    y_e, _ = self.attn_experts[e](x_e, x_e, x_e, attn_mask=causal_mask)
                    slot_out[e_mask] = y_e
            expert_outs.append(slot_out)

        # --- 3. weighted sum & dropout --------------------------------
        # stack: (k, B, S, D) → (B, k, S, D)
        stacked = torch.stack(expert_outs, dim=1)
        weighted = (stacked * probs[:, :, None, None]).sum(dim=1)  # (B,S,D)
        return self.dropout(weighted), self.load_balance_loss()


class HeadSwitchSelfAttention(nn.Module):
    """
    Multi‑Head Self‑Attention where *each head* has `num_experts`
    value/output projections.
    For every token we pick the Top‑k experts **independently per head**.
    The expensive QK soft‑max is done **once**; only the much cheaper
    V‑proj & out‑proj are routed.

    Parameters
    ----------
    d_model      : hidden size
    n_head       : number of logical heads (h)
    num_experts  : experts per head (E)
    k            : top‑k experts to activate **per head**  (usually 1)
    """

    def __init__(
            self,
            d_model: int,
            n_head: int,
            num_experts: int,
            k: int = 1,
            dropout: float = 0.1,
            ema_decay: float = 0.99,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.h = n_head
        self.d_h = d_model // n_head
        self.E = num_experts
        self.k = k

        # shared Q,K projections and bias‑less gate
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

        # ------------------------------------------------------------------
        # Each *head* owns a [E × d_h × d_h] tensor of V and O weights
        # They are registered as Parameters so they are sharded correctly.
        # Shape: (h, E, d_h, d_h)
        self.W_v = nn.Parameter(torch.empty(n_head, num_experts, self.d_h, self.d_h))
        self.W_o = nn.Parameter(torch.empty(n_head, num_experts, self.d_h, self.d_h))
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_o)

        # Gating: per head, per token  –  logits (B,S,h,E)
        self.router = nn.Linear(d_model, n_head * num_experts, bias=False)

        # Stats for load‑balancing
        self.register_buffer("_ema_counts", torch.zeros(n_head, num_experts))
        self.ema_decay = ema_decay
        self.dropout = nn.Dropout(dropout)

    # ------------- optional auxiliary loss ------------------------------
    def load_balance_loss(self):
        probs = self._ema_counts / (self._ema_counts.sum() + 1e-9)
        return (probs * probs).sum() * self.E * self.h

    # --------------------------------------------------------------------

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x    : (B, S, D)
        mask : (S, S) additive – already ‑inf where causal
        """
        B, S, D = x.shape
        # ---- 1. shared Q, K, attention weights -------------------------
        q = self.W_q(x).view(B, S, self.h, self.d_h).transpose(1, 2)  # (B,h,S,d_h)
        k = self.W_k(x).view(B, S, self.h, self.d_h).transpose(1, 2)  # (B,h,S,d_h)

        attn_scores = torch.einsum("bhsd,bhtd->bhst", q, k) / math.sqrt(self.d_h)
        attn_scores = attn_scores + mask  # broadcast over batch/h
        p_attn = F.softmax(attn_scores, dim=-1)  # (B,h,S,S)

        # ---- 2. router decides V/O expert per head ---------------------
        gate_logits = self.router(x)  # (B,S,h*E)
        gate_logits = gate_logits.view(B, S, self.h, self.E)
        topk_scores, topk_idx = gate_logits.topk(self.k, dim=-1)  # (B,S,h,k)

        # soft probabilities for mixture‑of‑experts
        probs = F.softmax(topk_scores, dim=-1)  # (B,S,h,k)

        # track EMA of choices   (hard, per head)
        # topk_idx : (B, S, h, k)
        with torch.no_grad():
            # one‑hot encode *without* flattening the head dim
            # shape → (B, S, h, k, E)
            oh = F.one_hot(topk_idx, num_classes=self.E).float()

            # sum over batch, sequence, and k slots, keep (h,E)
            hard_counts = oh.sum(dim=(0, 1, 3))  # (h, E)

            # exponential moving average
            denom = B * S * self.k  # tokens that cast a vote
            self._ema_counts.mul_(self.ema_decay) \
                .add_(hard_counts / denom,
                      alpha=1.0 - self.ema_decay)

        # ---- 3. compute expert V/O projections only for chosen experts -
        # Pre‑compute x reshaped once to (B,S,h,d_h) for all heads
        x_h = x.view(B, S, self.h, self.d_h)

        # v_out = torch.zeros_like(x_h)  # (B,S,h,d_h)
        # o_out = torch.zeros_like(x_h)

        v_out = torch.zeros(self.h, B, S, self.d_h, device=x.device, dtype=x.dtype)
        o_out = torch.zeros(self.h, B, S, self.d_h, device=x.device, dtype=x.dtype)

        # loop over k (tiny, usually 1)
        for slot in range(self.k):
            idx_e = topk_idx[..., slot]  # (B,S,h)
            p_e = probs[..., slot]  # (B,S,h)

            # idx_e : (B, S, h)   →  (h, B, S)
            idx_e_perm = idx_e.permute(2, 0, 1)  # reorder so head is leading

            # head indices 0..h‑1, one per plane, broadcast across B and S
            head_idx = torch.arange(self.h, device=idx_e.device) \
                .view(self.h, 1, 1).expand(self.h, B, S)  # (h, B, S)

            # Now the two indexing tensors have identical shape → OK
            W_v_e = self.W_v[head_idx, idx_e_perm]  # (h, B, S, d_h, d_h)
            W_o_e = self.W_o[head_idx, idx_e_perm]  # same shape

            # move head to batch dim for matmul: (h*B*S, d_h)
            tokens = x_h.permute(2, 0, 1, 3).reshape(-1, self.d_h)  # (h*B*S, d_h)
            v_proj = torch.bmm(tokens.unsqueeze(1),  # (h*B*S,1,d_h)
                               W_v_e.reshape(-1, self.d_h, self.d_h)) \
                .squeeze(1).view(self.h, B, S, self.d_h)  # back to (h,B,S,d_h)

            # attention aggregation (uses shared weights)
            attn_out = torch.einsum("bhst,hbsd->hbt d", p_attn, v_proj)  # (h,B,S,d_h)

            # output projection
            o_proj = torch.bmm(attn_out.reshape(-1, 1, self.d_h),
                               W_o_e.reshape(-1, self.d_h, self.d_h)) \
                .squeeze(1).view(self.h, B, S, self.d_h)

            # p_e : (B, S, h)
            weight = p_e.unsqueeze(-1)  # → (B, S, h, 1)   ✔ aligns with B,S,h

            v_out += (attn_out.permute(1, 2, 0, 3) * weight) \
                .permute(2, 0, 1, 3)  # back to (h, B, S, d_h)

            o_out += (o_proj.permute(1, 2, 0, 3) * weight) \
                .permute(2, 0, 1, 3)

        # ---- 4. put heads back together, dropout -----------------------
        y = o_out.permute(1, 2, 0, 3).reshape(B, S, D)
        return self.dropout(y), self.load_balance_loss()


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            d_ff: int,
            num_ff_experts: int,
            num_attn_experts: int = 1,  # ← new
            top_k: int = 2,
            dropout: float = 0.1,
            max_seq_len: int = 256,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)

        if num_attn_experts > 1:
            self.self_attn = HeadSwitchSelfAttention(
                d_model, n_head, num_attn_experts, dropout=dropout
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, n_head, dropout=dropout, batch_first=True
            )

        self.ln2 = nn.LayerNorm(d_model)
        self.moe_ff = MoEFeedForward(d_model, d_ff, num_ff_experts)
        self.dropout = nn.Dropout(dropout)
        causal = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")), 1
        )
        self.register_buffer("_causal_mask", causal, persistent=False)
        self.top_k = top_k

    def forward(self, x, top_k=None):
        S = x.size(1)
        mask = self._causal_mask[:S, :S]
        # --- attention ------------------------------------------------
        if isinstance(self.self_attn, nn.MultiheadAttention):
            sa_out, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
            aux_attn = 0.0
        else:
            sa_out, aux_attn = self.self_attn(x, mask)

        x = x + self.dropout(sa_out)
        x = self.ln1(x)

        # --- ffn -------------------------------------------------------
        ff_out, aux_ffn = self.moe_ff(x, k=self.top_k)
        x = x + self.dropout(ff_out)
        x = self.ln2(x)
        return x, (aux_attn + aux_ffn)

    def expert_token_stats(self):
        stats = {}

        # For MoE Self-Attention (if present and provides similar stats)
        if hasattr(self, 'self_attn') and hasattr(self.self_attn, '_total_tokens'):
            # Check if self_attn appears to be an MoE layer with the expected attributes
            stats["attn_total"] = int(self.self_attn._total_tokens)
            if hasattr(self.self_attn, '_token_counts'):
                stats["attn_by_exp"] = self.self_attn._token_counts.cpu().tolist()
            if hasattr(self.self_attn, '_gate_selection_counts'):
                stats[
                    "attn_gate_selection_by_exp"] = self.self_attn._gate_selection_counts.cpu().tolist()

        # For MoE FeedForward layer (self.moe_ff)
        if hasattr(self, 'moe_ff'):  # Should always exist based on your structure
            stats["ffn_total"] = int(self.moe_ff._total_tokens)
            stats["ffn_by_exp"] = self.moe_ff._token_counts.cpu().tolist()  # Actual routed tokens
            # Add the pre-capacity gate selection counts
            stats["ffn_gate_selection_by_exp"] = self.moe_ff._gate_selection_counts.cpu().tolist()

        return stats


class MoETransformerLM(nn.Module):
    """A minimal GPT‑style language model with Mixture‑of‑Experts FFNs."""

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            n_head: int = 4,
            num_layers: int = 1,
            d_ff: int = 1024,
            num_experts: int = 4,
            num_attn_experts: int = 4,
            top_k: int = 2,
            layer_repetition: int = 4,
            num_initial_layers: int = 2,
            num_final_layers: int = 2,
            max_seq_len: int = 256,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.initial_layers = nn.ModuleList(TransformerBlock(d_model, n_head, d_ff,
                                                             num_ff_experts=num_experts,
                                                             num_attn_experts=num_attn_experts,
                                                             dropout=dropout,
                                                             top_k=top_k,
                                                             max_seq_len=max_seq_len) for _ in
                                            range(num_initial_layers))
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                n_head,
                d_ff,
                num_ff_experts=num_experts,
                num_attn_experts=num_attn_experts,
                dropout=dropout,
                top_k=top_k,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        )
        self.final_layers = nn.ModuleList(TransformerBlock(d_model, n_head, d_ff,
                                                          num_ff_experts=num_experts,
                                                          num_attn_experts=num_attn_experts,
                                                          dropout=dropout,
                                                          top_k=top_k,
                                                          max_seq_len=max_seq_len) for _ in range(num_final_layers))
        self.layer_repetition = layer_repetition

        self.repeat_emb = RepeatEmbedding(self.layer_repetition, d_model)

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

        for init_layer in self.initial_layers:
            x, aux = init_layer(x)
            aux_losses = aux_losses + aux
        for rep in range(self.layer_repetition):
            x = self.repeat_emb(rep, x)
            for layer in self.layers:
                x, aux = layer(x)
                aux_losses = aux_losses + aux

        for final_layer in self.final_layers:
            x, aux = final_layer(x)
            aux_losses = aux_losses + aux

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, aux_losses

    def token_statistics(self) -> dict:
        """
        Returns
        -------
        stats : dict
            {
              "total"                          : int,
              "attn_by_exp"                    : list[int] | None,
              "attn_gate_selection_by_exp"     : list[int] | None, # NEW
              "ffn_by_exp"                     : list[int],
              "ffn_gate_selection_by_exp"      : list[int],        # NEW
              "layers"         : [
                  {
                    "name"                         : str,
                    "attn_total"                   : int,
                    "attn_by_exp"                  : list[int] | None,
                    "attn_gate_selection_by_exp"   : list[int] | None, # NEW
                    "ffn_total"                    : int,
                    "ffn_by_exp"                   : list[int],
                    "ffn_gate_selection_by_exp"    : list[int],        # NEW
                  },
                  # ...
              ],
            }
        """

        # ------------------------------------------------------------------
        # A helper that merges element‑wise counts (used for the aggregates)
        def _merge(dst, src):
            if src is None:
                return dst
            if dst is None:
                # Ensure to copy if src is not None, to avoid modifying original list from bstats
                return list(
                    src)  # Changed to list(src) for safety, was src.copy() which might fail if not list

            # If dst exists, extend it if necessary and add values
            for i, v in enumerate(src):
                if i >= len(dst):
                    dst.append(v)
                else:
                    dst[i] += v
            return dst

        # ------------------------------------------------------------------

        blocks = [(f"init{i}", blk) for i, blk in enumerate(self.initial_layers)] + \
                 [(f"L{i}", blk) for i, blk in enumerate(self.layers)] + \
                 [(f"final{i}", blk) for i, blk in enumerate(self.final_layers)]

        # True “how many tokens have passed through the model?”
        # This assumes _total_tokens in the first layer's FFN MoE represents this.
        # Ensure this layer exists and has the attribute.
        true_total = 0
        if self.initial_layers and hasattr(self.initial_layers[0], 'moe_ff') and \
                hasattr(self.initial_layers[0].moe_ff, '_total_tokens'):
            true_total = int(self.initial_layers[0].moe_ff._total_tokens)

        agg_attn_by_exp, agg_ffn_by_exp = None, None
        agg_attn_gate_selection, agg_ffn_gate_selection = None, None  # NEW aggregates

        per_layer_stats: list[dict] = []  # Renamed for clarity

        for name, blk in blocks:
            bstats = blk.expert_token_stats()  # local stats for this block

            # Prepare stats for this layer
            layer_detail = {
                "name": name,
                "attn_total": bstats.get("attn_total", 0),
                "attn_by_exp": bstats.get("attn_by_exp"),
                "attn_gate_selection_by_exp": bstats.get("attn_gate_selection_by_exp"),  # NEW
                "ffn_total": bstats.get("ffn_total", 0),  # Use .get for ffn_total for safety
                "ffn_by_exp": bstats.get("ffn_by_exp"),  # Use .get for ffn_by_exp
            }
            # ffn_gate_selection_by_exp should be present if ffn_by_exp is
            if "ffn_by_exp" in bstats:  # Check if ffn stats are present
                layer_detail["ffn_gate_selection_by_exp"] = bstats.get(
                    "ffn_gate_selection_by_exp")  # NEW

            per_layer_stats.append(layer_detail)

            # Aggregate stats
            agg_attn_by_exp = _merge(agg_attn_by_exp, bstats.get("attn_by_exp"))
            agg_ffn_by_exp = _merge(agg_ffn_by_exp, bstats.get("ffn_by_exp"))
            agg_attn_gate_selection = _merge(agg_attn_gate_selection,
                                             bstats.get("attn_gate_selection_by_exp"))  # NEW
            # Ensure ffn_gate_selection_by_exp is merged only if present
            if "ffn_by_exp" in bstats:  # Condition this on ffn MoE stats being available
                agg_ffn_gate_selection = _merge(agg_ffn_gate_selection,
                                                bstats.get("ffn_gate_selection_by_exp"))  # NEW

        return {
            "total": true_total,
            "attn_by_exp": agg_attn_by_exp,
            "attn_gate_selection_by_exp": agg_attn_gate_selection,  # NEW
            "ffn_by_exp": agg_ffn_by_exp,
            "ffn_gate_selection_by_exp": agg_ffn_gate_selection,  # NEW
            "layers": per_layer_stats,
        }
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
