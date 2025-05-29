import torch
import torch.nn as nn


class TitansPersistentMemory(nn.Module):
    """Fixed, learnable tokens that are prepended to every segment.
    They are *not* updated at test‑time. Acts like the paper’s “persistent
    memory” (§3.3)."""

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return the persistent tokens expanded to the current batch."""
        return self.mem.unsqueeze(0).expand(batch_size, -1, -1)


class NeuralMemory(nn.Module):
    """Deep neural long‑term memory (§3.1) with a fast key/value store.

    This implementation keeps *num_slots* key/value vectors that are
    continuously refreshed with a simple write‑on‑surprise rule.  The
    write path is parameterised (an MLP) so the block can *learn how* to
    store information, while the read path is attention over keys.

    The true Titans memory trains its weights online with momentum and a
    decay gate; here we provide a minimal approximation that is easy to
    extend. Surprise can be any scalar signal (e.g. loss or grad‑norm).
    """

    def __init__(self, dim: int, hidden_dim: int, num_slots: int, *,
                 temp: float | None = None, gamma: float = 0.99):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.register_buffer("keys", torch.randn(num_slots, dim) * 0.02)
        self.register_buffer("values", torch.randn(num_slots, dim) * 0.02)

        # Learnable write projection (deep memory: 2‑layer MLP).
        self.write_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)
        )
        self.temp = temp or dim ** -0.5  # read temperature
        self.gamma = gamma              # exponential decay / forgetting rate
        self._cursor = 0                # circular pointer for replacement

    @torch.no_grad()
    def update(self, x: torch.Tensor, surprise: torch.Tensor | None = None):
        """Write *x* (B,L,D) into memory with exponential decay.

        This is *one* plausible instantiation of Eq.(13–14) from the
        paper: we first decay every slot by *gamma* and then overwrite a
        single slot chosen in round‑robin fashion.  Replace this logic to
        experiment with other gating or optimisation‑based rules.
        """
        new_key = x.mean(dim=1).mean(dim=0)           # very coarse key
        new_val = self.write_proj(new_key)            # deep write path

        # decay / forget (Eq.13)
        self.keys.mul_(self.gamma)
        self.values.mul_(self.gamma)

        # write (Eq.14) – simple replacement, but could be surprise‑gated
        idx = self._cursor % self.num_slots
        self.keys[idx].copy_(new_key)
        self.values[idx].copy_(new_val)
        self._cursor += 1

    def retrieve(self, q: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
        """Attention read: q (B,L,D) → (B,L,D)."""
        sim = torch.einsum("bld,nd->bln", q, self.keys) * self.temp
        if top_k is not None and top_k < self.num_slots:
            sim_top, idx = torch.topk(sim, k=top_k, dim=-1)
            attn = sim_top.softmax(dim=-1)
            v = self.values[idx]                       # (B,L,k,D)
            out = (attn.unsqueeze(-1) * v).sum(dim=-2) # (B,L,D)
        else:
            attn = sim.softmax(dim=-1)
            out = torch.einsum("bln,nd->bld", attn, self.values)
        return out

    def forward(self, q: torch.Tensor, *, write_x: torch.Tensor | None = None,
                surprise: torch.Tensor | None = None) -> torch.Tensor:
        out = self.retrieve(q)
        if write_x is not None:
            self.update(write_x, surprise=surprise)
        return out


class SlidingWindowSelfAttention(nn.Module):
    """Core attention (short‑term memory) limited to *window* tokens.

    Any windowed/efficient attention could be plugged here; we keep it
    simple with `torch.nn.MultiheadAttention` + an explicit causal mask
    so the time complexity is still O(window²), but constant w.r.t. the
    *full* sequence length used by the Titans MAC design.
    """

    def __init__(self, dim: int, num_heads: int, window: int):
        super().__init__()
        self.dim = dim
        self.window = window
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.register_buffer("_mask", self._build_mask(window), persistent=False)

    def _build_mask(self, size: int) -> torch.Tensor:
        mask = torch.full((size, size), float("-inf"))
        for i in range(size):
            start = max(0, i - self.window + 1)
            mask[i, start:i + 1] = 0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if self._mask.size(0) < L:
            self._mask = self._build_mask(L).to(x.device)
        return self.attn(x, x, x, attn_mask=self._mask[:L, :L])[0]


class TitansMACBlock(nn.Module):
    """Memory‑as‑Context block (Fig.2 & §4.1).

    Steps (Eq.21–25):
      1. Retrieve *historical* memory corresponding to the current
         segment (queries = x) from the neural memory.
      2. Concatenate persistent tokens and retrieved memory with x.
      3. Run short‑range attention on this augmented sequence.
      4. Update the neural memory with the segment.
      5. Return the slice that corresponds to the original tokens.

    The block is plug‑and‑play: stack several to build a Titan.
    """

    def __init__(self, dim: int, num_heads: int, window: int,
                 *, num_persistent_tokens: int = 4, memory_slots: int = 128,
                 mem_hidden_dim: int | None = None):
        super().__init__()
        mem_hidden_dim = mem_hidden_dim or (4 * dim)
        self.persistent = TitansPersistentMemory(num_persistent_tokens, dim)
        self.memory = NeuralMemory(dim, mem_hidden_dim, memory_slots)
        self.core_attn = SlidingWindowSelfAttention(dim, num_heads, window)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # (1) retrieve historical context
        mem_ctx = self.memory(x, write_x=None)          # (B,T,D)

        # (2) add persistent tokens
        p = self.persistent(B)                          # (B,P,D)
        aug = torch.cat([p, mem_ctx, x], dim=1)         # (B,P+T+T,D)

        # (3) core short‑term attention
        h = self.core_attn(aug)
        h = self.ffn(self.norm(h)) + h                  # simple residual

        # (4) update long‑term memory with *current* segment
        self.memory.update(x)

        # (5) return only the outputs aligned with the original tokens
        return h[:, -T:]


# Small sanity check
if __name__ == "__main__":
    torch.manual_seed(0)
    blk = TitansMACBlock(dim=64, num_heads=4, window=128)
    dummy = torch.randn(2, 64, 64)  # (B, T, D)
    out = blk(dummy)
    print(out.shape)  # torch.Size([2,64,64])

import torch
import torch.nn as nn
import torch.nn.functional as F


class TitansPersistentMemory(nn.Module):
    """Fixed, learnable tokens that are prepended to every segment.
    They are *not* updated at test‑time. Acts like the paper’s “persistent
    memory” (§3.3)."""

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return the persistent tokens expanded to the current batch."""
        return self.mem.unsqueeze(0).expand(batch_size, -1, -1)


class NeuralMemory(nn.Module):
    """
    Neural Long-Term Memory (LTM) that learns to memorize at test time
    by updating its own parameters.
    Based on concepts from Behrouz, Zhong, and Mirrokni "Titans" paper.
    """

    def __init__(self,
                 query_dim: int,  # Dimension of input queries to the LTM (from x_t or S^(t))
                 key_dim: int,  # Dimension of keys k_t for associative loss
                 value_dim: int,  # Dimension of values v_t for associative loss
                 memory_net_hidden_dim: int,
                 # Hidden dimension of the internal MLP acting as memory
                 memory_net_layers: int = 2,  # Number of layers in the internal MLP
                 learning_rate: float = 0.01,  # LR for updating memory_net parameters
                 alpha_forget: float = 0.01,  # Forgetting factor (1-alpha_t) * M_t-1
                 theta_surprise_lr: float = 0.01,  # Factor for momentary surprise
                 eta_momentum: float = 0.9  # Momentum for surprise signal S_t
                 ):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim  # This is D_in for k_t = x_t W_K
        self.value_dim = value_dim  # This is D_in for v_t = x_t W_V
        # and also output dim of memory_net M(k_t)

        # The LTM itself is a neural network (e.g., an MLP) [cite: 130]
        # Its parameters will be updated at test time.
        # It maps keys k_t to values v_t: M(k_t) -> v_t
        layers = []
        current_dim = self.key_dim
        for _ in range(memory_net_layers - 1):
            layers.append(nn.Linear(current_dim, memory_net_hidden_dim))
            layers.append(nn.ReLU())  # Or SiLU as per paper's architectural details [cite: 233]
            current_dim = memory_net_hidden_dim
        layers.append(nn.Linear(current_dim, self.value_dim))
        self.memory_net = nn.Sequential(*layers)

        # Optimizer for the memory_net's parameters
        # This is part of the "inner-loop" optimization [cite: 125]
        self.optimizer = torch.optim.SGD(self.memory_net.parameters(), lr=learning_rate)

        # Projections for forming k_t, v_t from input (e.g., y_t from MAC block attention)
        # These are W_K and W_V from Eq. 11[cite: 122].
        # Their parameters are learned in the "outer-loop" (i.e., standard model training).
        # Assuming input_to_kv_proj_dim is the dimension of y_t (attention output)
        # For simplicity, let's assume it's query_dim for now, but it should be dim of y_t.
        input_to_kv_proj_dim = self.query_dim  # Placeholder: This should be dim of y_t
        self.key_projection_for_loss = nn.Linear(input_to_kv_proj_dim, self.key_dim)
        self.value_projection_for_loss = nn.Linear(input_to_kv_proj_dim, self.value_dim)

        # For surprise mechanism S_t = eta_t * S_t-1 - theta_t * grad(loss) [cite: 114]
        # S_t here will represent the "momentum" of parameter updates.
        # We'll store it as a list of tensors, matching memory_net.parameters().
        self.S_momentum = [torch.zeros_like(p) for p in self.memory_net.parameters()]

        self.alpha_forget = alpha_forget  # Corresponds to (1-alpha_t) factor for M_t-1, paper uses alpha_t for decay part [cite: 128]
        # So this should be (1 - actual_alpha_forget_gate_value)
        self.theta_surprise_lr = theta_surprise_lr
        self.eta_momentum = eta_momentum

        # Note: The paper's alpha_t, eta_t, theta_t are data-dependent [cite: 116, 118]
        # For simplicity, we are using fixed scalars here.

    def _associative_loss(self, k_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        """ Computes the associative memory loss: ||M(k_t) - v_t||^2_2 (Eq. 12) """
        predicted_v_t = self.memory_net(k_t)
        loss = F.mse_loss(predicted_v_t, v_t)
        return loss

    # @torch.enable_grad() # Ensure gradients are enabled for LTM update
    def update(self, content_to_memorize: torch.Tensor, surprise_gate: torch.Tensor):
        """
        Updates the memory_net parameters based on content_to_memorize and surprise.
        content_to_memorize: (B, L, D_y) - e.g., y_t from MAC attention block.
        surprise_gate: (B,) - scalar gating signal for each batch item.
                         In the paper, surprise is more complex (Eq. 10 [cite: 114]).
                         Here, we use it as a simple gate for now.
        """
        B, L, D_y = content_to_memorize.shape

        # This update should happen "at test time" [cite: 93]
        # For a batch, we might average gradients or handle them per item.
        # Paper implies online updates per x_t (or segment S^(t)).
        # Let's average over the sequence length L for simplicity here.

        # Project content to k_t_batch, v_t_batch for the loss
        # These are k_t = x_t W_K, v_t = x_t W_V (Eq. 11) [cite: 122]
        # Here, x_t is 'content_to_memorize' (which should be y_t from MAC).
        k_t_batch = self.key_projection_for_loss(content_to_memorize)  # (B, L, key_dim)
        v_t_batch = self.value_projection_for_loss(content_to_memorize)  # (B, L, value_dim)

        # Detach k_t and v_t from the main computation graph if these projections
        # are also part of the outer model that shouldn't get gradients from LTM update.
        k_t_batch = k_t_batch.detach()
        v_t_batch = v_t_batch.detach()

        for i in range(B):
            if surprise_gate[i].item() > 0.5:  # Simplified surprise gating
                # For tokens in the segment
                # The paper processes tokens x_t within a segment S^(t) [cite: 197]
                # For simplicity, let's average the loss over the sequence.
                # A more faithful implementation would iterate or process vectorized.

                current_k_t = k_t_batch[i]  # (L, key_dim)
                current_v_t = v_t_batch[i]  # (L, value_dim)

                # Enable gradients for memory_net parameter update
                for p in self.memory_net.parameters():
                    p.grad = None  # Clear old gradients

                # Compute loss (Eq. 12) [cite: 123]
                # Need to retain graph for .backward() if memory_net params require grad
                original_training_states = {}
                for name, param in self.memory_net.named_parameters():
                    original_training_states[name] = param.requires_grad
                    param.requires_grad_(True)

                loss = self._associative_loss(current_k_t, current_v_t)

                # Compute gradients for M_t-1: nabla l(M_t-1; x_t)
                loss.backward()

                # Update memory_net parameters using the surprise mechanism (Eq. 13, 14) [cite: 128]
                with torch.no_grad():  # We manually update weights
                    for idx, p in enumerate(self.memory_net.parameters()):
                        if p.grad is not None:
                            # Momentary surprise: -theta_t * grad(loss)
                            momentary_surprise_grad = -self.theta_surprise_lr * p.grad

                            # Update S_t = eta_t * S_t-1 + momentary_surprise_grad [cite: 114]
                            # (paper uses '-' for grad, I used '+' assuming S_t is the update step)
                            # Let's re-align: S_t = eta*S_{t-1} - theta*grad(l)
                            # Then M_t = (1-alpha)*M_{t-1} + S_t (if S_t is the full update step)
                            # OR M_t = (1-alpha)*M_{t-1} - S_t (if S_t is momentum like Adam)
                            # Paper: M_t = (1-alpha_t)M_{t-1} + S_t, where S_t = eta*S_{t-1} - theta*grad(l)

                            self.S_momentum[idx] = self.eta_momentum * self.S_momentum[idx] + \
                                                   momentary_surprise_grad  # This is -theta*grad

                            # Apply forgetting to M_t-1 and add the surprise update S_t
                            # M_t = (1-alpha_t)M_{t-1} + S_t
                            # Here self.alpha_forget is actually (1-alpha_t_paper)
                            p.data.mul_(1.0 - self.alpha_forget)  # M_t-1 * (1 - alpha_gate_value)
                            p.data.add_(self.S_momentum[
                                            idx])  # Add the full S_t (momentum + current grad step)

                # Restore original requires_grad state
                for name, param in self.memory_net.named_parameters():
                    param.requires_grad_(original_training_states[name])

    def retrieve(self, query_input: torch.Tensor) -> torch.Tensor:
        """
        Retrieves information from the LTM using the query_input.
        query_input: (B, L_query, D_query) - This is q_t (e.g., S^(t)W_Q) [cite: 199]
        Output: (B, L_query, value_dim) - This is M*(q_t) (Eq. 15) [cite: 142]
        """
        # Ensure memory_net is in eval mode if it has dropout/batchnorm,
        # though for simple MLPs it might not matter.
        # self.memory_net.eval() # Potentially

        # The paper doesn't specify that q_t for retrieval (Eq. 15) is the same as
        # k_t for the associative loss (Eq. 12).
        # q_t is x_t W_Q. k_t for loss is x_t W_K.
        # Here, query_input is assumed to be already projected (q_t).
        # The memory_net itself maps from key_dim to value_dim.
        # So, the query_input for M* should be of key_dim.
        # This implies query_input (q_t) needs to be projected to key_dim if it's not already.
        # Let's assume query_input has query_dim, and we need a projection to key_dim.
        # For now, IF query_dim == key_dim, we can pass it directly.
        # This is a slight ambiguity resolution based on M(k_t) structure.
        # However, Eq. 15 M*(q_t) suggests M takes q_t directly.
        # If M's first layer is from key_dim, then q_t must be key_dim.

        if query_input.shape[-1] != self.memory_net[0].in_features:
            # This would happen if W_Q projected x_t to a dimension different from W_K's output
            # This requires a clear definition of input dimension for memory_net
            # For now, let's assume query_input has self.key_dim
            # This means W_Q (used externally to produce query_input) projects to self.key_dim
            pass

        # Reshape for batch processing by memory_net if L_query > 1
        B, L_query, D_q = query_input.shape
        query_input_flat = query_input.reshape(B * L_query, D_q)

        retrieved_values_flat = self.memory_net(query_input_flat)

        retrieved_values = retrieved_values_flat.reshape(B, L_query, self.value_dim)
        return retrieved_values

    def forward(self,
                query_for_retrieve: torch.Tensor,
                *,
                content_to_memorize: torch.Tensor | None = None,
                surprise_gate: torch.Tensor | None = None) -> torch.Tensor:
        """
        Performs retrieval and, if content is provided, updates the memory.
        The order (retrieve then update, or vice-versa) depends on the overall model flow.
        Fig 2: Retrieval from M_t-1, then Attention, then Update to get M_t.
        So, retrieval uses the "current" state of memory_net.
        Then, if update is called, memory_net becomes M_t for the *next* step.
        """
        retrieved_context = self.retrieve(query_for_retrieve)

        if content_to_memorize is not None and surprise_gate is not None:
            # This `update` call modifies `self.memory_net` in place, effectively
            # transitioning M_t-1 to M_t for subsequent operations in the next timestep/segment.
            self.update(content_to_memorize, surprise_gate=surprise_gate)

        return retrieved_context


class SlidingWindowSelfAttention(nn.Module):
    """Core attention (short‑term memory) limited to *window* tokens.
    Uses a causal mask within the window."""

    def __init__(self, dim: int, num_heads: int, window: int):
        super().__init__()
        self.dim = dim
        self.window = window
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # Causal mask for the window
        self.register_buffer("_mask", self._build_causal_mask(window), persistent=False)

    def _build_causal_mask(self, size: int) -> torch.Tensor:
        # Standard causal mask: True means masked
        mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
        return mask

    def _get_mask(self, L: int, device: torch.device) -> torch.Tensor:
        if self._mask.size(0) < L or self._mask.device != device:
            self._mask = self._build_causal_mask(L).to(device)
        return self._mask[:L, :L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # For sliding window, the mask needs to be more complex if not using full causal.
        # The original implementation's mask was more like a fixed-width band.
        # For simplicity and common use, nn.MultiheadAttention's attn_mask is often full (L,L).
        # If a true sliding window (each token attends to W previous) is needed without
        # processing segments, it requires careful masking or custom attention.
        # The current causal mask means each token attends to itself and all previous tokens *within the input x*.
        # If x is already a segment of length `window`, this is fine.

        # Let's assume x is the full augmented sequence for this layer, and
        # `window` implies a conceptual limit, but nn.MHA will use a full mask over L.
        # If `window` is strictly enforced for compute, typically sequence is chunked.
        # The provided `_build_mask` in the original code was for a fixed-size window.
        # Let's stick to a simpler causal mask over the given sequence length L.
        # If `window` is meant to be a hard constraint on attention span,
        # the input `x` to this module should already be chunked to that `window` size.
        # Or, a more complex mask is needed.
        # The original `_build_mask` was:
        #   mask = torch.full((size, size), float("-inf"))
        #   for i in range(size):
        #       start = max(0, i - self.window + 1)
        #       mask[i, start:i + 1] = 0
        # This mask is (L,L) where 0 is attend, -inf is don't.
        # nn.MultiheadAttention expects True for masked positions if attn_mask is bool.

        # Reverting to a mask similar to the original intent for fixed window size
        # This mask allows attending to `window` previous tokens + current.
        # True means masked.
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        for i in range(L):
            start = max(0, i - self.window + 1)
            mask[i, start: i + 1] = False  # Allow attention to these

        # Ensure causal if window > L
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        final_mask = mask | causal_mask  # Combine window constraint with causality

        return self.attn(x, x, x, attn_mask=final_mask)


class TitansMACBlock(nn.Module):
    """Memory‑as‑Context block (Fig.2 & §4.1).

    Steps (Eq.21–25 from paper, adapted):
      1. Retrieve *historical* memory corresponding to the current
         segment (queries = x) from the neural memory.
      2. Concatenate persistent tokens and retrieved memory with x.
      3. Run short‑range attention on this augmented sequence.
      4. Generate content for LTM update from attention output.
      5. Update the neural memory with this content, gated by surprise.
      6. Return the slice that corresponds to the original tokens.
    """

    def __init__(self, dim: int, num_heads: int, window: int,
                 *, num_persistent_tokens: int = 4, memory_slots: int = 128,
                 mem_hidden_dim: int | None = None, surprise_threshold: float = 0.1):
        super().__init__()
        mem_hidden_dim = mem_hidden_dim or (4 * dim)  # Default for LTM's internal MLP
        self.persistent_mem = TitansPersistentMemory(num_persistent_tokens,
                                                     dim)  # Renamed for clarity
        self.neural_mem = NeuralMemory(dim, mem_hidden_dim, memory_slots,
                                       surprise_threshold=surprise_threshold)  # Renamed for clarity
        self.core_attn = SlidingWindowSelfAttention(dim, num_heads, window)

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)  # Norm after FFN

    def forward(self, x: torch.Tensor, surprise_signal: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        P = self.persistent_mem.mem.shape  # Number of persistent tokens

        # (1) Retrieve historical context from NeuralMemory using current input x as query
        # write_x is None here, so memory is only read, not updated yet.
        mem_ctx = self.neural_mem(x, write_x=None, surprise_signal=None)  # (B, T, D)

        # (2) Concatenate persistent tokens, retrieved LTM context, and original input x
        persistent_tokens = self.persistent_mem(B)  # (B, P, D)

        # The MAC diagram often shows M_p, M_ltm, x_t.
        # Here mem_ctx is derived from x, so it's like M_ltm(x).
        # Concatenation order: Persistent, LTM context, current input
        aug_input = torch.cat([persistent_tokens, mem_ctx, x], dim=1)  # (B, P + T + T, D)

        # (3) Core short‑term attention + FFN (standard Transformer block structure)
        # Apply LayerNorm before attention
        h_attn = self.core_attn(self.norm1(aug_input))

        # Residual connection for attention part
        # The input to attention was aug_input, so residual is with aug_input
        h_residual_attn = aug_input + h_attn  # Or use a projection if dims change

        # FFN part
        h_ffn = self.ffn(self.norm2(h_residual_attn))

        # Residual connection for FFN part
        h_final = h_residual_attn + h_ffn  # (B, P + T + T, D)

        # (4) Prepare content for LTM update from the attention output
        # We use the part of 'h_final' that corresponds to the original input tokens 'x'.
        # This creates the feedback loop: attention output informs memory.
        # The slice corresponding to x was the last T tokens of aug_input.
        # So, the corresponding output is the last T tokens of h_final.
        content_for_ltm_update = h_final  # (B, T, D)

        # (5) Update long‑term memory with this content
        # If no surprise_signal is provided, we might use a default or skip update.
        # For this example, let's make surprise_signal mandatory if we want updates.
        if surprise_signal is not None:
            self.neural_mem.update(content_for_ltm_update, surprise_signal)
        # Else, LTM is not updated in this forward pass.

        # (6) Return only the outputs aligned with the original input tokens 'x'
        # This is the slice of h_final corresponding to the original 'x' positions.
        output = h_final  # (B, T, D)

        return output


# Small sanity check
if __name__ == "__main__":
    torch.manual_seed(0)

    dim = 64
    num_heads = 4
    window = 32  # Max attention span for core_attn
    batch_size = 2
    seq_len = 64  # Length of input sequence x
    num_persistent_tokens = 4
    memory_slots = 128

    print("--- Initializing TitansMACBlock ---")
    blk = TitansMACBlock(dim=dim, num_heads=num_heads, window=window,
                         num_persistent_tokens=num_persistent_tokens,
                         memory_slots=memory_slots,
                         surprise_threshold=0.05)  # Example threshold

    dummy_x = torch.randn(batch_size, seq_len, dim)  # (B, T, D)

    # Simulate a surprise signal (e.g., from a loss calculation)
    # For this test, let's make one batch item surprising, the other not.
    dummy_surprise = torch.tensor([0.5, 0.01], dtype=torch.float).view(batch_size, 1)  # (B,1)

    print(f"\n--- Input x shape: {dummy_x.shape} ---")
    print(
        f"--- Dummy surprise signal: {dummy_surprise.squeeze().tolist()} (threshold: {blk.neural_mem.surprise_threshold}) ---")

    print("\n--- Neural Memory content BEFORE first pass (first 2 slots) ---")
    print("Keys:", blk.neural_mem.keys[:2])
    print("Values:", blk.neural_mem.values[:2])
    print("Cursor:", blk.neural_mem._cursor)

    # First pass
    out1 = blk(dummy_x, surprise_signal=dummy_surprise)
    print(f"\n--- Output 1 shape: {out1.shape} ---")  # Should be (B, T, D)

    print("\n--- Neural Memory content AFTER first pass (first 2 slots) ---")
    print("Keys:", blk.neural_mem.keys[:2])  # Should be updated for the surprising item
    print("Values:", blk.neural_mem.values[:2])
    print("Cursor:", blk.neural_mem._cursor)  # Should have advanced if surprise was met

    # Second pass with different surprise
    dummy_x_2 = torch.randn(batch_size, seq_len, dim)
    dummy_surprise_2 = torch.tensor([0.02, 0.6], dtype=torch.float).view(batch_size, 1)
    print(f"\n--- Input x_2 shape: {dummy_x_2.shape} ---")
    print(f"--- Dummy surprise signal 2: {dummy_surprise_2.squeeze().tolist()} ---")

    out2 = blk(dummy_x_2, surprise_signal=dummy_surprise_2)
    print(f"\n--- Output 2 shape: {out2.shape} ---")

    print("\n--- Neural Memory content AFTER second pass (first 2 slots) ---")
    print("Keys:", blk.neural_mem.keys[:2])  # Should be further updated
    print("Values:", blk.neural_mem.values[:2])
    print("Cursor:", blk.neural_mem._cursor)

    # Test retrieval logic in NeuralMemory
    print("\n--- Testing NeuralMemory retrieval ---")
    nm = NeuralMemory(dim=dim, hidden_dim=dim * 2, num_slots=10, surprise_threshold=0.1)
    # Manually set some keys/values for predictable retrieval
    nm.keys = torch.ones(dim)
    nm.values = torch.ones(dim) * 5
    nm.keys[1] = torch.ones(dim) * -1
    nm.values[1] = torch.ones(dim) * -5

    query = torch.ones(1, 1, dim)  # Query that should be similar to key
    retrieved_val = nm.retrieve(query)
    print("Query:", query[0, 0, :4].tolist())
    print("Retrieved value (first 4 dims):",
          retrieved_val[0, 0, :4].tolist())  # Expecting something close to 5

    query_orthogonal = torch.randn(1, 1, dim)
    query_orthogonal[:, :, 0] = 100  # Make it different
    retrieved_val_orth = nm.retrieve(query_orthogonal)
    print("Orthogonal Query (first 4 dims):", query_orthogonal[0, 0, :4].tolist())
    print("Retrieved value for orthogonal query (first 4 dims):",
          retrieved_val_orth[0, 0, :4].tolist())

    # Test top_k retrieval
    retrieved_top1 = nm.retrieve(query, top_k=1)
    print("Retrieved top_k=1 (first 4 dims):", retrieved_top1[0, 0, :4].tolist())
    assert torch.allclose(retrieved_top1, nm.values.view(1, 1, -1),
                          atol=1e-5)  # Should be exactly values if temp is right