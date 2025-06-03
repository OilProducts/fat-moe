from math import sqrt


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any

# ──────────────────────────────────────────────────────────────────────────
# 1.  Utilities copied from previous snippets
# ──────────────────────────────────────────────────────────────────────────
def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1)          # [B,S]

def entropy_segments(ent: torch.Tensor) -> torch.Tensor:
    inc  = F.pad((ent[:, 1:] > ent[:, :-1]).int(), (1, 0))
    seg_id = torch.cumsum(inc, dim=1)                  # [B,S]
    return seg_id

def segment_attn_mask(seg_id: torch.Tensor,
                      num_queries: int,
                      num_heads: int) -> torch.Tensor:
    """Return bool mask shaped (B*num_heads, Q, S) where True ⇒ BLOCK."""
    B, S = seg_id.shape
    Q = num_queries
    device = seg_id.device

    same = seg_id[:, None, :] == torch.arange(Q, device=device)[None, :, None]
    mask = ~same                                         # True = block
    mask = mask.repeat_interleave(num_heads, dim=0)      # (B*H,Q,S)
    return mask


def build_segment_queries_mask(
    seg_id: torch.Tensor,          # [B, S]  int
    query_embed: torch.Tensor,     # [L, D]  (learned parameters)
    num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        queries  : [B, S*L, D]
        att_mask : [B*num_heads, S*L, S]   (True ⇒ block)
        valid    : [B, S]  bool   – True where segment actually exists
    """
    B, S = seg_id.shape
    L, D = query_embed.shape
    device = seg_id.device

    seg_count = seg_id.max(dim=1).values + 1             # [B]
    S_hat     = seg_count.max().item()                   # max segments in batch

    # ------- (1)  queries ------------------------------------------------
    #   repeat learned queries for every possible segment index
    queries = (query_embed
                .unsqueeze(0)            # (1, L, D)
                .unsqueeze(0)            # (1, 1, L, D)
                .repeat(B, S_hat, 1, 1))   # (B, S, L, D)
    queries = queries.view(B, S_hat*L, D)                # (B, Q_tot, D)

    # ------- (2)  segment id for each query q ---------------------------
    seg_for_q = torch.arange(S_hat, device=device)       \
                    .repeat_interleave(L)                # (Q_tot,)
    seg_for_q = seg_for_q.unsqueeze(0).expand(B, -1)     # (B, Q_tot)

    # ------- (3)  attention mask ----------------------------------------
    same   = seg_for_q[:, :, None] == seg_id[:, None, :] # (B, Q_tot, S)
    att_mask = ~same                                     # True ⇒ block
    att_mask = att_mask.repeat_interleave(num_heads, 0)  # (B*H, Q_tot, S)

    # ------- (4)  validity map (optional) -------------------------------
    valid = torch.arange(S_hat, device=device)[None, :] < seg_count[:, None]  # (B,Ŝ)

    return queries, att_mask, valid

def safe_softmax(scores: torch.Tensor,
                 mask: torch.Tensor,             # same shape, True ⇒ block
                 dim: int = -1) -> torch.Tensor:
    """
    Identical to soft‑max on `scores.masked_fill(mask, -inf)` *except*
    rows where everything is masked yield a zero vector (no NaNs).
    """
    scores = scores.masked_fill(mask, float("-inf"))

    # Identify rows that are completely −inf
    all_masked = torch.isneginf(scores).all(dim=dim, keepdim=True)

    # Replace −inf with 0 in such rows so exp() = 1 → softmax = 1/rowlen
    safe_scores = scores.masked_fill(all_masked, 0.0)

    attn = torch.softmax(safe_scores, dim=dim)

    # Bring the fully‑masked rows back to exact zeros
    attn = attn.masked_fill(all_masked, 0.0)
    return attn


# ──────────────────────────────────────────────────────────────────────────
# 2.  Vector‑Quantiser (same as previous, abbreviated doc‑string)
# ──────────────────────────────────────────────────────────────────────────
class VectorQuantiser(nn.Module):
    def __init__(self, K: int, D: int, beta: float = 0.25,
                 ema: bool = True, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.K, self.D, self.beta, self.ema = K, D, beta, ema
        self.decay, self.eps = decay, eps
        self.codebook = nn.Parameter(torch.randn(K, D))
        if ema:
            self.register_buffer("ema_cluster_size", torch.zeros(K))
            self.register_buffer("ema_weight_sum", self.codebook.data.clone())

    @torch.no_grad()
    def _ema_update(self, encodings, flat):
        # encodings : (N,K) one‑hot
        # flat      : (N,D)
        dw = encodings.T @ flat  # (K,D)
        cluster_size = encodings.sum(0)  # (K,)

        # Exponential moving average
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_weight_sum.mul_(self.decay).add_(dw, alpha=1 - self.decay)

        # --- normalise so their sum equals the true token count -------------
        n = self.ema_cluster_size.sum()
        ema_cluster_size = ((self.ema_cluster_size + self.eps) /
                            (n + self.K * self.eps)) * n  # ← missing step

        # Final code‑book update
        self.codebook.data.copy_(self.ema_weight_sum / ema_cluster_size.unsqueeze(1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, Q, D = z.shape
        flat = z.reshape(-1, D)                               # (N,D)
        dist = (flat.pow(2).sum(1, True)
                - 2*flat @ self.codebook.T
                + self.codebook.pow(2).sum(1))             # (N,K)
        idx = dist.argmin(1)                               # (N,)
        z_q = self.codebook[idx].view(B, Q, D)

        # losses
        commit = F.mse_loss(z, z_q.detach())
        codebk = F.mse_loss(z_q, z.detach())
        vq_loss = self.beta*commit + codebk

        # straight‑through estimator
        z_q = z + (z_q - z).detach()

        if self.training and self.ema:
            enc = F.one_hot(idx, self.K).type_as(flat)     # (N,K)
            self._ema_update(enc, flat)

        return z_q, vq_loss, idx.view(B, Q)

# ──────────────────────────────────────────────────────────────────────────
# 3.  Learned‑query pooler (unchanged except for attn_mask passthrough)
# # ──────────────────────────────────────────────────────────────────────────
# class LearnedQueryAttention(nn.Module):
#     def __init__(self, dim: int, num_queries: int, num_heads: int,
#                  dropout: float = 0.0):
#         super().__init__()
#         if dim % num_heads:
#             raise ValueError("embed_dim must be divisible by num_heads")
#         self.num_queries, self.num_heads = num_queries, num_heads
#         self.query = nn.Parameter(torch.randn(num_queries, dim))
#         self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
#                                          batch_first=True)
#
#     def forward(self,
#                 x: torch.Tensor,
#                 queries: Optional[torch.Tensor] = None,
#                 attn_mask: Optional[torch.Tensor] = None,
#                 key_padding_mask: Optional[torch.Tensor] = None
#                ) -> Tuple[torch.Tensor, torch.Tensor]:
#         return self.mha(query=queries,
#                         key=x, value=x,
#                         attn_mask=attn_mask,
#                         key_padding_mask=key_padding_mask)

class LearnedQueryAttention(nn.Module):
    """
    Multi‑Head Attention where *queries* are L learned vectors reused for every
    segment.  Masking is handled outside and passed in as `attn_mask`.
    """
    def __init__(self,
                 embed_dim: int,
                 num_queries_per_segment: int,
                 num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.d_model  = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.L         = num_queries_per_segment

        self.in_norm = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)

        # Learned query template  – (L, D)
        self.query_template = nn.Parameter(torch.randn(self.L, embed_dim))

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.query_template, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                      # (B, S, D)   keys & values
        queries: torch.Tensor,                # (B, Q_tot, D)  pre‑built set
        attn_mask: torch.Tensor,              # (B*H, Q_tot, S)  bool
        key_padding_mask: Optional[torch.Tensor] = None    # (B,S)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            attn_output  : (B, Q_tot, D)
            attn_weights : (B, Q_tot, S)  (optional; detach for logging)
        """
        B, S, _ = x.shape
        Q_tot   = queries.size(1)

        x = self.in_norm(x)

        # project
        q = self.q_proj(queries)              # (B, Q_tot, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split heads ---------------------------------------------------
        q = q.view(B, Q_tot, self.num_heads, self.head_dim) \
              .permute(0, 2, 1, 3)            # (B, H, Q, d_h)
        k = k.view(B, S,     self.num_heads, self.head_dim) \
              .permute(0, 2, 3, 1)            # (B, H, d_h, S)
        v = v.view(B, S,     self.num_heads, self.head_dim) \
              .permute(0, 2, 1, 3)            # (B, H, S, d_h)

        # scaled dot‑product -------------------------------------------
        scores = (q @ k) / sqrt(self.head_dim)          # (B, H, Q, S)

        # combine masks -------------------------------------------------
        mask = attn_mask.view(B, self.num_heads, Q_tot, S)
        if key_padding_mask is not None:
            mask = mask | key_padding_mask[:, None, None, :]

        attn = safe_softmax(scores, mask, dim=-1)       # (B, H, Q, S)
        attn = self.dropout(attn)

        # weighted sum --------------------------------------------------
        out = attn @ v                                  # (B, H, Q, d_h)
        out = out.permute(0, 2, 1, 3).reshape(B, Q_tot, self.d_model)
        out = self.out_norm(out)

        return self.out_proj(out), attn.mean(dim=1)     # (B,Q,D), (B,Q,S)


# ──────────────────────────────────────────────────────────────────────────
# 4.  Sliding‑window Transformer encoder (same as earlier but returns both
#     hidden states *and* logits so we keep entropy computation local)
# ──────────────────────────────────────────────────────────────────────────

class SlidingWindowAttention(nn.Module):
    """Multi‑head scaled‑dot‑product attention restricted to a fixed
    retrospective *window* of tokens.

    Each position *i* in the sequence is allowed to attend only to the range
    [max(0, i−window_size), …, i].  Future tokens and tokens further in the
    past are masked out.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension of the model.
    num_heads : int
        Number of attention heads.
    window_size : int
        How many *previous* tokens (inclusive) a query may attend to.
    dropout : float, optional
        Drop‑out applied to the attention probabilities.
    bias : bool, optional
        Whether the projection layers use bias terms.

    Notes
    -----
    •  The implementation builds an explicit (seq_len, seq_len) boolean mask.
    •  The layer is causal by design (no look‑ahead).
    """

    def __init__(self, embed_dim: int, num_heads: int, window_size: int,
                 dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    # ---------------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------------
    def _sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return a (seq_len, seq_len) boolean mask for the current window."""
        idx = torch.arange(seq_len, device=device)
        rel_pos = idx[None, :] - idx[:, None]          # (seq_len, seq_len)
        mask = (rel_pos > 0) | (rel_pos < -self.window_size)
        return mask  # True where attention should be *blocked*

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute sliding‑window attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        attn_mask : torch.Tensor, optional
            Additional mask broadcast‑compatible with (seq_len, seq_len).  Should
            be *True* (or 1) where attention is **not permitted**.
        key_padding_mask : torch.Tensor, optional
            Boolean mask of shape (batch, seq_len) marking **padding** tokens.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        B, S, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, H, S, D)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = q * (self.head_dim ** -0.5)
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, H, S, S)

        mask = self._sliding_window_mask(S, x.device)
        if attn_mask is not None:
            mask = mask | attn_mask.to(torch.bool)
        if key_padding_mask is not None:
            # Broadcast (B, 1, 1, S)
            expanded_kpm = key_padding_mask[:, None, None, :]
            mask = mask.unsqueeze(0) | expanded_kpm
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)  # (B, H, S, D)
        y = y.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        y = self.out_proj(y)
        return y

class SlidingWindowTransformer(nn.Module):
    def __init__(self, dim, heads, window, vocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.attn = SlidingWindowAttention(dim, heads, window)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim), nn.ReLU(), nn.Linear(4*dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, vocab)

    def forward(self, x):                         # x: (B,S)
        h = self.embedding(x)
        h = h + self.attn(self.norm1(h))
        h = h + self.ffn(self.norm2(h))
        logits = self.proj(self.norm3(h))                     # (B,S,V)
        return h, logits                          # hidden + logits

# ──────────────────────────────────────────────────────────────────────────
# 5.  The full byte‑sequence compressor
# ──────────────────────────────────────────────────────────────────────────
class ByteSegmentCompressor(nn.Module):
    """
    End‑to‑end module that turns a sequence of byte‑level tokens into
    a *shorter* sequence of:
      • continuous vectors   (segment embeddings)
      • discrete indices     (code‑book look‑ups)

    Output dict:
      {
        'continuous' : (B, Q_max, D)   # pooled segment vectors (quantised)
        'codes'      : (B, Q_max)      # int64 indices into code‑book
        'vq_loss'    : scalar tensor   # 0 if VQ disabled / eval
      }
    """
    def __init__(self,
                 vocab_size: int = 259,           # 0‑255 + specials
                 dim: int = 256,
                 heads: int = 8,
                 window: int = 128,
                 num_queries: int = 1,
                 codebook_size: int = 512,
                 beta: float = 0.25):
        super().__init__()
        self.encoder = SlidingWindowTransformer(dim, heads, window, vocab_size)
        self.pooler  = LearnedQueryAttention(dim, num_queries, heads)
        self.vq      = VectorQuantiser(codebook_size, dim, beta)

    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
        """
        token_ids : (B, S)  int16/32
        key_padding_mask (optional) : (B,S) bool
        """
        # ── 1.  Encoder ────────────────────────────────────────────────
        hidden, logits = self.encoder(token_ids)        # (B,S,D) , (B,S,V)

        # ── 2.  Entropy‑based segmentation ─────────────────────────────
        with torch.no_grad():
            entropy   = token_entropy(logits)               # (B,S)
            seg_id = entropy_segments(entropy)               # (B,S)
        # mask = segment_attn_mask(segid,
        #                          self.pooler.num_queries,
        #                          self.pooler.num_heads)  # (B*H,Q,S)

        queries, seg_mask, valid = build_segment_queries_mask(
            seg_id,
            self.pooler.query_template,  # learned [L,D] parameter tensor
            self.pooler.num_heads
        )


        # ── 3.  Learned‑query pooling restricted per segment ───────────
        pooled, _ = self.pooler(hidden,
                                queries=queries,
                                attn_mask=seg_mask,
                                key_padding_mask=key_padding_mask)  # (B,Q,D)

        # ── 4.  Vector‑quantise ────────────────────────────────────────
        quantised, vq_loss, codes = self.vq(pooled)     # (B,Q,D) , scalar , (B,Q)

        return {'continuous': quantised,
                'codes'     : codes,
                'vq_loss'   : vq_loss,
                'valid_mask': valid}


class CodeExpander(nn.Module):
    """
    Converts a sequence of high‑level codes (vocab K_hi, length L_hi)
    into a longer sequence of low‑level codes (vocab K_lo, length L_lo).

    • Training  – teacher forcing with (x_hi, y_lo) pairs
    • Inference – autoregressive generation until EOS (or max_len)

    Parameters
    ----------
    K_hi, K_lo : int          vocab sizes of the two levels
    D          : int          model dimension
    N_enc      : int          #encoder layers
    N_dec      : int          #decoder layers
    H          : int          attention heads
    """

    def __init__(self,
                 K_hi: int,
                 K_lo: int,
                 D: int = 256,
                 N_enc: int = 4,
                 N_dec: int = 4,
                 H: int = 8,
                 dropout: float = 0.1,
                 eos_id: int = 1,          # reserve 0 for PAD if you like
                 max_len: int = 2048):
        super().__init__()
        self.K_hi, self.K_lo = K_hi, K_lo
        self.D, self.eos_id, self.max_len = D, eos_id, max_len

        # ── Embeddings & positional encodings ───────────────────────────
        self.emb_hi = nn.Embedding(K_hi, D)
        self.emb_lo = nn.Embedding(K_lo, D)
        self.pos_enc = nn.Parameter(torch.randn(max_len, D))  # learned PE

        # ── Encoder & Decoder stacks ───────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(D, H, 4*D, dropout, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(D, H, 4*D, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, N_enc)
        self.decoder = nn.TransformerDecoder(dec_layer, N_dec)

        # ── Output projection to low‑level vocab ───────────────────────
        self.out_proj = nn.Linear(D, K_lo)

    # ------------------------------------------------------------------
    #  Helper: build causal mask for the decoder
    # ------------------------------------------------------------------
    def _causal_mask(self, L: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), 1)

    # ------------------------------------------------------------------
    #  Forward (training / teacher forcing)
    # ------------------------------------------------------------------
    def forward(self,
                codes_hi: torch.Tensor,           # (B, L_hi)
                codes_lo: torch.Tensor            # (B, L_lo)  ground truth
               ) -> Dict[str, torch.Tensor]:
        B, L_hi = codes_hi.shape
        _, L_lo = codes_lo.shape
        device = codes_hi.device

        # Encoder
        enc = self.emb_hi(codes_hi) + self.pos_enc[:L_hi]      # (B,L_hi,D)
        enc = self.encoder(enc)                                # (B,L_hi,D)

        # Decoder input (shift‑right)
        dec_inp = F.pad(codes_lo[:, :-1], (1,0), value=self.eos_id)  # BOS = EOS
        dec = self.emb_lo(dec_inp) + self.pos_enc[:L_lo]

        tgt_mask = self._causal_mask(L_lo, device)             # (L_lo,L_lo)

        dec = self.decoder(tgt=dec,
                           memory=enc,
                           tgt_mask=tgt_mask)

        logits = self.out_proj(dec)                            # (B,L_lo,K_lo)
        return {'logits': logits}

    # ------------------------------------------------------------------
    #  Inference  (autoregressive)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self,
                 codes_hi: torch.Tensor,                       # (B,L_hi)
                 max_len: Optional[int] = None
                ) -> torch.Tensor:                             # (B, ≤max_len)
        B, L_hi = codes_hi.shape
        device = codes_hi.device
        max_len = max_len or self.max_len

        enc = self.emb_hi(codes_hi) + self.pos_enc[:L_hi]
        enc = self.encoder(enc)

        generated = torch.full((B, 1), self.eos_id, device=device, dtype=torch.long)

        for t in range(max_len):
            dec = self.emb_lo(generated) + self.pos_enc[:generated.size(1)]
            tgt_mask = self._causal_mask(dec.size(1), device)
            dec = self.decoder(tgt=dec, memory=enc, tgt_mask=tgt_mask)
            next_logits = self.out_proj(dec[:, -1])            # (B, K_lo)
            next_token  = next_logits.argmax(-1, keepdim=True) # greedy
            generated   = torch.cat([generated, next_token], dim=1)
            if (next_token == self.eos_id).all():
                break
        return generated[:, 1:]   # drop initial BOS/EOS token


class StackedByteCompressor(nn.Module):
    """
    Four‑level hierarchy that repeatedly compresses its input.
      level‑0 : bytes  (259‑symbol vocab)
      level‑1 : codes from level‑0   (K‑symbol vocab)
      level‑2 : codes from level‑1   (K‑symbol vocab)
      level‑3 : codes from level‑2   (K‑symbol vocab)

    Args
    ----
    dim                : embedding dimension per level (same for all)
    heads              : attention heads per level
    window             : sliding window per level
    num_queries        : max segments per sample per level
    codebook_size_mult : multiplier for codebook size (K = codebook_size_mult * 512)
    beta               : commitment cost for every VQ layer
    """

    def __init__(self,
                 dim: int = 256,
                 heads: int = 8,
                 window: int = 128,
                 num_queries: int = 64,
                 codebook_size_mult: int = 512,
                 beta: float = 0.25):
        super().__init__()

        self.levels: List[ByteSegmentCompressor] = nn.ModuleList()

        # ── Level‑0 operates on raw bytes (259 tokens) ────────────────
        codebook_size = 259
        self.levels.append(
            ByteSegmentCompressor(vocab_size=codebook_size,
                                  dim=dim, heads=heads, window=window,
                                  num_queries=num_queries,
                                  codebook_size=codebook_size * codebook_size_mult,
                                  beta=beta)
        )

        # ── Levels 1‑3 operate on previous code‑book indices ─────────
        for _ in range(3):
            codebook_size = codebook_size_mult * 512
            self.levels.append(
                ByteSegmentCompressor(vocab_size=codebook_size,
                                      dim=dim, heads=heads, window=window,
                                      num_queries=num_queries,
                                      codebook_size=codebook_size,
                                      beta=beta)
            )

    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        tokens : (B, S₀)  – raw byte IDs  (0‑258)

        Returns a dict with lists of tensors, one per level:
          'continuous' : [(B,Q,D), …]   length=4
          'codes'      : [(B,Q), …]     length=4
          'vq_loss'    : scalar          (sum of all levels)
        """
        cont_list: List[torch.Tensor] = []
        code_list: List[torch.Tensor] = []
        total_vq  = torch.tensor(0., device=tokens.device)

        inp = tokens                        # start with bytes
        for lvl, compressor in enumerate(self.levels):
            out = compressor(inp)
            cont_list.append(out['continuous'])
            code_list.append(out['codes'])
            total_vq = total_vq + out['vq_loss']
            # discrete codes of this level become input for the next
            inp = out['codes']

        return {'continuous': cont_list,
                'codes'     : code_list,
                'vq_loss'   : total_vq}


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

    def encode(self, text: str, add_bos: bool | None = None, add_eos: bool | None = None) -> list[
        int]:
        """
        Convert text to a list of integers where each character is its
        raw byte (0-255). Optionally add BOS/EOS tokens outside 0-255 range.
        """
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        # Convert directly to bytes (0..255)
        raw_bytes = text.encode("utf-8", errors="ignore")

        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend(raw_bytes)  # These will be 0..255
        if add_eos:
            tokens.append(self.eos_id)

        return torch.tensor(tokens, dtype=torch.int16)

    def decode(self, tokens: list[int], cut_at_eos: bool = False) -> str:
        """
        Reconstruct text by ignoring any token >= 256 and turning the rest
        (0..255) back into bytes. If `cut_at_eos=True`, stop at the first EOS.
        """
        # Optionally cut at EOS
        if cut_at_eos and (self.eos_id in tokens):
            # slice up to and including EOS (or you can slice before if desired)
            eos_index = tokens.index(self.eos_id)
            tokens = tokens[: eos_index + 1]

        # Filter out special tokens >= 256
        byte_list = [t for t in tokens if t < 256]

        # Decode to UTF-8 string
        return bytes(byte_list).decode("utf-8", errors="ignore")

    def get_token_offsets(self, text: str, tokens: list[int] | None = None):
        """
        (Optional) If you need a way to map tokens back to character offsets in the text,
        you can implement it here. Not strictly necessary for a minimal reproducing example.
        """
        raise NotImplementedError()


class HierarchicalAutoencoder(nn.Module):
    """
    Encapsulates a stack of ByteSegmentCompressors and CodeExpanders
    for hierarchical compression and decompression of byte sequences.
    """

    def __init__(self,
                 num_levels: int,
                 # Compressor configurations (one dict per level)
                 # Each dict should contain args for ByteSegmentCompressor:
                 # {'dim': int, 'heads': int, 'window': int, 'num_queries': int, 'codebook_size': int, 'beta': float}
                 # The 'vocab_size' for the first compressor is initial_vocab_size.
                 # For subsequent compressors, 'vocab_size' is the 'codebook_size' of the previous.
                 compressor_level_configs: List[Dict[str, Any]],
                 initial_vocab_size: int = 259,  # For the very first level (bytes)
                 # Expander base configuration (shared across expander levels)
                 expander_dim_scale: float = 1.0,  # Multiplier for compressor dim to get expander dim
                 expander_num_enc_layers: int = 4,
                 expander_num_dec_layers: int = 4,
                 expander_heads_scale: float = 1.0,  # Multiplier for compressor heads
                 expander_dropout: float = 0.1,
                 expander_eos_id: int = 1,  # As used in CodeExpander for BOS padding
                 expander_max_len: int = 2048,  # Default max generation length
                 propagate_key_padding_mask: bool = True  # If True, ByteSegmentCompressor must return 'valid_mask'
                 ):
        super().__init__()

        if len(compressor_level_configs) != num_levels:
            raise ValueError("Length of compressor_level_configs must match num_levels.")

        self.num_levels = num_levels
        self.initial_vocab_size = initial_vocab_size
        self.expander_eos_id = expander_eos_id
        self.propagate_key_padding_mask = propagate_key_padding_mask

        # ---- Configure Compressor Stack ----
        self.compressors = nn.ModuleList()
        current_input_vocab_size = initial_vocab_size
        self.actual_codebook_sizes: List[int] = []  # Stores K_i for compressor_i's output

        for i in range(num_levels):
            config = compressor_level_configs[i]
            compressor = ByteSegmentCompressor(
                vocab_size=current_input_vocab_size,
                dim=config['dim'],
                heads=config['heads'],
                window=config['window'],
                num_queries=config['num_queries'],
                codebook_size=config['codebook_size'],
                beta=config['beta']
            )
            self.compressors.append(compressor)
            self.actual_codebook_sizes.append(config['codebook_size'])
            current_input_vocab_size = config['codebook_size']  # Output of this is input to next

        # ---- Configure Expander Stack ----
        # Expanders work in reverse order of compression.
        # self.expanders[0] takes top_codes (output of self.compressors[-1])
        # self.expanders[-1] reconstructs original byte tokens.
        self.expanders = nn.ModuleList()

        for i in range(num_levels - 1, -1, -1):
            # This expander reconstructs the input to self.compressors[i].
            # Its K_hi (input codes) are the output of self.compressors[i].
            # Its K_lo (output codes) are the input vocab of self.compressors[i].

            k_hi = self.actual_codebook_sizes[i]
            base_compressor_config = compressor_level_configs[i]

            if i == 0:  # This expander reconstructs the original byte tokens
                k_lo = self.initial_vocab_size
            else:  # This expander reconstructs codes from the previous compression level
                k_lo = self.actual_codebook_sizes[i - 1]

            exp_dim = int(base_compressor_config['dim'] * expander_dim_scale)
            exp_heads = int(base_compressor_config['heads'] * expander_heads_scale)
            if exp_heads == 0: exp_heads = 1  # Ensure at least one head

            expander = CodeExpander(
                K_hi=k_hi,
                K_lo=k_lo,
                D=exp_dim,
                N_enc=expander_num_enc_layers,
                N_dec=expander_num_dec_layers,
                H=exp_heads,
                dropout=expander_dropout,
                eos_id=expander_eos_id,
                max_len=expander_max_len
            )
            self.expanders.append(expander)
        # self.expanders is now [TopExpander, ..., BottomExpander]

    def compress(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Applies the full stack of compressors.
        Args:
            tokens: Input byte tokens (B, S_initial).
            key_padding_mask: Optional boolean mask for `tokens` (B, S_initial), True indicates padding.
        Returns:
            A dictionary containing:
                'top_codes': Codes from the final compression level (B, S_top).
                'all_codes': List of code tensors from each compression level.
                             [codes_level_0, codes_level_1, ..., codes_level_N-1]
                'all_continuous': List of continuous (quantized) vectors from each level.
                'vq_loss': Sum of VQ losses from all levels.
                'final_key_padding_mask': Key padding mask for 'top_codes' if propagated.
        """
        all_codes_list: List[torch.Tensor] = []
        all_continuous_list: List[torch.Tensor] = []
        all_input_seq_lengths: List[int] = []
        all_output_seq_lengths: List[int] = []
        total_vq_loss = torch.tensor(0., device=tokens.device, dtype=torch.float)

        current_input_tokens = tokens
        current_kpm = key_padding_mask

        # Crucial: ByteSegmentCompressor's forward must return 'valid_mask' (True where segment exists)
        # for key_padding_mask propagation to work. 'valid_mask' is (B, Q_max).
        # The new key_padding_mask for the next level becomes `~valid_mask`.

        for i, compressor in enumerate(self.compressors):
            if current_kpm is not None:
                # Effective input length is non-padded length
                input_seq_len = (~current_kpm).sum(dim=1).float().mean().item() # Avg non-padded length
            else:
                input_seq_len = current_input_tokens.size(1)
            all_input_seq_lengths.append(int(input_seq_len))

            comp_out = compressor(current_input_tokens, key_padding_mask=current_kpm)
            output_codes = comp_out['codes']

            valid_mask_for_output = comp_out.get('valid_mask')
            if valid_mask_for_output is not None:
                # Effective output length is number of valid segments
                output_seq_len = valid_mask_for_output.sum(dim=1).float().mean().item() # Avg valid segments
            else:
                output_seq_len = output_codes.size(1)
            all_output_seq_lengths.append(int(output_seq_len))

            all_codes_list.append(output_codes)
            all_continuous_list.append(comp_out['continuous'])
            total_vq_loss += comp_out['vq_loss']

            current_input_tokens = output_codes
            if self.propagate_key_padding_mask:
                if valid_mask_for_output is not None:
                    current_kpm = ~valid_mask_for_output  # Next KPM: True where segment was NOT valid
                else:
                    if i < self.num_levels - 1:
                        print(
                            f"Warning: 'valid_mask' not returned by compressor {i}. KPM propagation for next level disabled.")
                    current_kpm = None
            else:
                current_kpm = None

        compression_ratios = [
            (out_len / in_len) if in_len > 0 else 0.0
            for in_len, out_len in zip(all_input_seq_lengths, all_output_seq_lengths)
        ]

        return {
            'top_codes': all_codes_list[-1] if all_codes_list else torch.empty(0, device=tokens.device),
            'all_codes': all_codes_list,
            'all_continuous': all_continuous_list,
            'vq_loss': total_vq_loss,
            'final_key_padding_mask': current_kpm,
            'compression_ratios': compression_ratios,  # Ratio of output/input effective lengths
            'input_seq_lengths': all_input_seq_lengths,  # Effective input lengths to each compressor
            'output_seq_lengths': all_output_seq_lengths  # Effective output lengths from each compressor
        }

    def decompress(self,
                   top_codes: torch.Tensor,
                   top_codes_key_padding_mask: Optional[torch.Tensor] = None,  # KPM for top_codes
                   targets_for_teacher_forcing: Optional[List[torch.Tensor]] = None,
                   # If teacher forcing: list of [codes_level_{N-2}, ..., codes_level_0, original_byte_tokens]
                   # These are the *target outputs* for each expander from top to bottom.
                   target_key_padding_masks: Optional[List[Optional[torch.Tensor]]] = None,  # KPMs for targets
                   max_len_override: Optional[int] = None
                   ) -> Dict[str, Any]:
        """
        Applies the full stack of expanders.
        Args:
            top_codes: Codes from the highest compression level (output of self.compressors[-1]).
            top_codes_key_padding_mask: Optional KPM for top_codes.
            targets_for_teacher_forcing: If provided, enables teacher forcing. List of target sequences
                                         for each expander from top down.
            target_key_padding_masks: Optional KPMs for each target sequence in teacher forcing.
            max_len_override: Override default max_len for generation.
        Returns:
            If training (teacher forcing):
                'all_logits': List of output logits from each expander.
                'final_reconstructed_logits': Logits for the original byte sequence.
            If inference:
                'generated_sequences': List of generated token sequences from each expander.
                'final_reconstructed_tokens': Generated byte tokens.
        """
        current_codes_to_expand = top_codes
        current_hi_kpm = top_codes_key_padding_mask  # KPM for codes_hi input to expander
        all_output_logits_list: List[torch.Tensor] = []
        all_generated_tokens_list: List[torch.Tensor] = []

        is_training = targets_for_teacher_forcing is not None
        if is_training and len(targets_for_teacher_forcing) != self.num_levels:
            raise ValueError("Length of targets_for_teacher_forcing must match num_levels.")
        if is_training and target_key_padding_masks and len(target_key_padding_masks) != self.num_levels:
            raise ValueError("Length of target_key_padding_masks must match num_levels if provided.")

        for i in range(self.num_levels):
            expander = self.expanders[i]  # TopExpander to BottomExpander

            if is_training:
                target_sequence_for_level = targets_for_teacher_forcing[i]
                tgt_kpm = target_key_padding_masks[i] if target_key_padding_masks else None

                # CodeExpander's forward needs: codes_hi, codes_lo, (optional masks)
                # It doesn't explicitly take key_padding_mask for codes_hi or codes_lo in its signature,
                # but TransformerEncoderLayer/DecoderLayer do.
                # For now, assume CodeExpander internally handles this or doesn't need explicit KPMs beyond target mask.
                # This part might need refinement if CodeExpander needs explicit KPMs for encoder/decoder inputs.
                exp_out = expander(codes_hi=current_codes_to_expand, codes_lo=target_sequence_for_level)
                all_output_logits_list.append(exp_out['logits'])

                # The input to the *next* expander (as codes_hi) is the *target output* of *this* expander.
                if i < self.num_levels - 1:
                    current_codes_to_expand = target_sequence_for_level
                    current_hi_kpm = tgt_kpm
            else:  # Inference
                # CodeExpander.generate might need src_key_padding_mask for its encoder part.
                # This is current_hi_kpm.
                # TODO: Modify CodeExpander.generate to accept src_key_padding_mask if necessary.
                # For now, assuming it's not critical or handled implicitly if short.
                generated_tokens = expander.generate(codes_hi=current_codes_to_expand,
                                                     max_len=max_len_override)
                all_generated_tokens_list.append(generated_tokens)
                current_codes_to_expand = generated_tokens
                current_hi_kpm = None  # KPM for generated sequences is typically not used for next step unless it's all EOS

        if is_training:
            return {
                'all_logits': all_output_logits_list,
                'final_reconstructed_logits': all_output_logits_list[-1]
            }
        else:
            return {
                'generated_sequences': all_generated_tokens_list,
                'final_reconstructed_tokens': all_generated_tokens_list[-1]
            }

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Full forward pass for training: compress, then decompress with teacher forcing, and compute losses.
        Args:
            tokens: Input byte tokens (B, S_initial).
            key_padding_mask: Optional KPM for input tokens.
        Returns:
            Dictionary with losses and final logits.
        """
        # 1. Compress
        compression_results = self.compress(tokens, key_padding_mask=key_padding_mask)
        # all_compressed_codes is [codes_L0, codes_L1, ..., codes_L(N-1)]
        all_compressed_codes = compression_results['all_codes']
        top_codes = compression_results['top_codes']  # This is codes_L(N-1)
        vq_loss = compression_results['vq_loss']

        # KPM for top_codes. If subsequent expanders use KPM for their `codes_hi` input.
        current_top_codes_kpm = compression_results['final_key_padding_mask']

        # --- Metrics for AIM ---
        compression_ratios = compression_results['compression_ratios']
        input_seq_lengths = compression_results['input_seq_lengths']
        output_seq_lengths = compression_results['output_seq_lengths']
        # --- End Metrics ---

        # 2. Prepare targets for decompression expander stack
        # self.expanders = [TopExp, Exp_N-2, ..., Exp_0 (BottomExp)]
        # Target for TopExp (self.expanders[0]) is input to compressor N-1 (which is codes_L(N-2))
        # Target for self.expanders[i] is input to compressor N-1-i
        #   which is codes_L(N-2-i)
        # Target for BottomExp (self.expanders[N-1]) is input to compressor 0 (original tokens)
        targets_for_expander_stack: List[torch.Tensor] = []
        # TODO: Prepare target_key_padding_masks for expander stack if propagate_key_padding_mask is True
        # This requires all_valid_masks from compressor stack.

        for i in range(self.num_levels - 1):  # Targets for expanders[0] to expanders[N-2]
            # Target for expanders[i] is all_compressed_codes[ (N-1) - 1 - i ]
            # = all_compressed_codes[ N - 2 - i ]
            targets_for_expander_stack.append(all_compressed_codes[self.num_levels - 2 - i])
        targets_for_expander_stack.append(tokens)  # Target for the last expander (bottom one)

        # 3. Decompress with teacher forcing
        decompression_results = self.decompress(
            top_codes=top_codes,
            top_codes_key_padding_mask=current_top_codes_kpm,
            targets_for_teacher_forcing=targets_for_expander_stack
            # target_key_padding_masks=... (needs to be built from compressor valid_masks)
        )
        all_reconstruction_logits = decompression_results['all_logits']

        # 4. Calculate losses
        total_reconstruction_loss = torch.tensor(0., device=tokens.device, dtype=torch.float)
        reconstruction_loss_details: Dict[str, torch.Tensor] = {}

        for i in range(self.num_levels):
            # all_reconstruction_logits[i] are predictions for targets_for_expander_stack[i]
            logits_i = all_reconstruction_logits[i]  # (B, S_target, K_target_vocab)
            target_i = targets_for_expander_stack[i]  # (B, S_target)

            # Assuming CodeExpander's logits and targets are aligned for direct CE.
            # Original training script used `logits[:,1:,:]` and `target[:,1:]`.
            # This implies ignoring the first token (e.g., a BOS or unconditioned first prediction).
            # For simplicity, let's use the whole sequence. If specific slicing is needed,
            # it can be done before passing to F.cross_entropy.
            # Or, ensure CodeExpander's output logits and target `codes_lo` match this.
            # If `CodeExpander.forward`'s `dec_inp` is padded (shifted right) from `codes_lo`,
            # then `logits` should have same seq_len as `codes_lo`.

            # Reshape for CrossEntropy: logits (N, C, ...), target (N, ...)
            # logits_i: (B, S, V) -> (B*S, V)
            # target_i: (B, S) -> (B*S)
            loss_i = F.cross_entropy(
                logits_i.reshape(-1, logits_i.size(-1)),
                target_i.reshape(-1),
                # ignore_index=... # TODO: Add pad_token_id if applicable
            )

            expander_k_lo = self.expanders[i].K_lo
            level_name = f"reconstruction_to_K{expander_k_lo}" + \
                         (
                             "_bytes" if expander_k_lo == self.initial_vocab_size else f"_codesL{self.num_levels - 2 - i if i < self.num_levels - 1 else '?'}")

            reconstruction_loss_details[level_name] = loss_i
            total_reconstruction_loss += loss_i

        # Average reconstruction loss over levels
        avg_reconstruction_loss = total_reconstruction_loss / self.num_levels if self.num_levels > 0 else torch.tensor(
            0.)
        final_total_loss = avg_reconstruction_loss + vq_loss

        return {
            'total_loss': final_total_loss,
            'vq_loss': vq_loss,
            'avg_reconstruction_loss': avg_reconstruction_loss,
            'reconstruction_loss_details': reconstruction_loss_details,
            'final_reconstructed_logits': decompression_results['final_reconstructed_logits'],
            # --- Metrics for AIM ---
            'compression_ratios': compression_ratios,
            'input_seq_lengths_compressors': input_seq_lengths,
            'output_seq_lengths_compressors': output_seq_lengths,
        }

    @torch.no_grad()
    def generate_bytes(self,
                       tokens: torch.Tensor,
                       key_padding_mask: Optional[torch.Tensor] = None,
                       max_len_override: Optional[int] = None) -> torch.Tensor:
        """
        Compresses input byte tokens and then autoregressively generates
        the reconstructed byte tokens.
        """
        self.eval()  # Ensure model is in evaluation mode (for dropout, etc.)
        compression_results = self.compress(tokens, key_padding_mask=key_padding_mask)
        top_codes = compression_results['top_codes']
        top_codes_kpm = compression_results['final_key_padding_mask']  # KPM for top_codes

        decompression_results = self.decompress(
            top_codes=top_codes,
            top_codes_key_padding_mask=top_codes_kpm,
            targets_for_teacher_forcing=None,  # Inference mode
            max_len_override=max_len_override
        )
        return decompression_results['final_reconstructed_tokens']


# ──────────────────────────────────────────────────────────────────────────
# 6.  Example usage
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, S = 2, 1024
    x = torch.randint(0, 259, (B, S))      # dummy byte tokens
    model = ByteSegmentCompressor()
    out = model(x)
    print("pooled shape :", out['continuous'].shape)   # (B, Q_max, D)
    print("codes shape  :", out['codes'].shape)        # (B, Q_max)
    print("vq_loss      :", out['vq_loss'].item())


# TODO: first sliding window transformer needs to take bytes, but after that we can take the continuous vectors