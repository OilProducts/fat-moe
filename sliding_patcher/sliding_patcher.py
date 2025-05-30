import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SlidingWindowAttention(nn.Module):
    """Multi‑head scaled‑dot‑product attention restricted to a fixed
    retrospective *window* of tokens.

    Each position *i* in the sequence is allowed to attend only to the range
    \[max(0, i−window_size), …, i].  Future tokens and tokens further in the
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
       For very long sequences you may want a block‑sparse or chunked
       implementation instead – let me know if you need that!
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

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:  # noqa: D401
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, window={self.window_size}"


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


class SlidingWindowTransformer(nn.Module):
    """
    A simple transformer model with sliding window attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, vocab_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_ff = embed_dim * 4  # Feed-forward dimension, can be adjusted

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = SlidingWindowAttention(embed_dim, num_heads, window_size)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, embed_dim)
        )
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.attention(x)  # Apply sliding window attention
        x = self.fc(x)
        x = self.out_proj(x)
        return x


def compute_last_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy (in nats) for the last token in the sequence.

    Args:
        logits: A tensor of shape [batch_size, seq_len, vocab_size].

    Returns:
        A tensor of shape [batch_size] with the entropy for the last token.
    """
    # Extract logits for the last token: shape [batch_size, vocab_size]
    last_token_logits = logits[:, -1, :]

    # Compute log probabilities
    log_probs = F.log_softmax(last_token_logits, dim=-1)  # [batch_size, vocab_size]

    # Convert to probabilities
    probs = torch.exp(log_probs)  # [batch_size, vocab_size]

    # Entropy: H = -sum_i p_i log p_i
    last_token_entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size]

    return last_token_entropy


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy (in nats) for every token in the sequence.

    Args
    ----
    logits : FloatTensor  [batch_size, seq_len, vocab_size]
        Raw, un‑normalised scores from the model.

    Returns
    -------
    entropies : FloatTensor  [batch_size, seq_len]
        Entropy of the predictive distribution at each position.
    """
    # Log‑probs for each token position
    log_probs = F.log_softmax(logits, dim=-1)          # [B, S, V]

    # Convert to probabilities on‑the‑fly and accumulate
    entropies = -(log_probs.exp() * log_probs).sum(dim=-1)   # [B, S]

    return entropies


def entropy_segments(ent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given per‑token entropy, produce:
      • segment_id  : [B, S]  int   – monotone‑decreasing runs
      • group_mask  : [B, S, S] bool – True where attention must be BLOCKED
    A new segment starts whenever entropy rises versus the previous token.
    """
    B, S = ent.shape
    # ‑‑ 1 if entropy increases, else 0; pad a zero up‑front
    inc = F.pad((ent[:, 1:] > ent[:, :-1]).int(), (1, 0))
    seg_id = torch.cumsum(inc, dim=1)                 # [B, S]

    # same‑segment ↔ allowed; different‑segment ↔ must be masked
    same = seg_id[:, :, None] == seg_id[:, None, :]   # [B, S, S] bool
    group_mask = ~same                                # True = block
    return seg_id, group_mask


def segment_attn_mask(
    seg_id: torch.Tensor,      # [B, S]  int  (0,1,2,… per segment)
    num_queries: int,
    num_heads: int
) -> torch.Tensor:
    """
    Returns `attn_mask` for nn.MultiheadAttention (batch_first=True).

    • Each learned query *q* is allowed to look ONLY at tokens whose
      segment_id == q; everything else is masked.

    Shape required by MHA (PyTorch ≥1.13):
        bool[ B*num_heads , num_queries , S ]
    """
    B, S = seg_id.shape
    device = seg_id.device
    Q = num_queries

    # Mask logic:  True  ⇒ block
    mask = torch.ones((B, Q, S), dtype=torch.bool, device=device)

    # For every batch b and position s, enable the matching query
    b_idx = torch.arange(B, device=device)[:, None]        # (B,1)
    s_idx = torch.arange(S, device=device)[None, :]        # (1,S)
    q_idx = seg_id                                         # (B,S)
    keep = q_idx[:, None, :] == torch.arange(Q, device=device)[None, :, None]
    mask = ~keep                                           # invert: True = block

    # Expand to per‑head mask that MultiheadAttention wants
    mask = mask.repeat_interleave(num_heads, dim=0)        # (B*H, Q, S)
    return mask



class LearnedQueryAttention(nn.Module):
    """
    Performs Multi-Head Attention where the queries are learned parameters
    and the keys/values are derived from an input sequence.

    This effectively computes a fixed-size set of output features by attending
    to the input sequence using learned points of focus.
    """
    def __init__(self,
                 embed_dim: int,
                 num_queries: int,
                 num_heads: int,
                 dropout: float = 0.0, # Dropout within MHA is often sufficient
                 device: Optional[torch.device | str] = None):
        """
        Initializes the LearnedQueryAttention module.

        Args:
            embed_dim: The embedding dimension (d_model).
            num_queries: The number of learned query vectors, determining the
                         output sequence length.
            num_heads: The number of attention heads. Must divide embed_dim.
            dropout: Dropout probability for the attention mechanism.
                     Note: nn.MultiheadAttention applies dropout internally.
            device: The device ('cpu', 'cuda', etc.) for parameters.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.device = device

        # Learned query parameters
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Multi-Head Attention layer
        # We use the standard MHA layer, leveraging its optimized implementation
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         batch_first=True, # Expects (batch, seq, feature)
                                         device=device)

        self._reset_parameters() # Initialize learned queries

    def _reset_parameters(self):
        # Initialize learned queries (can use Xavier/Kaiming or simple Gaussian)
        nn.init.normal_(self.query_embed, std=0.02) # Simple initialization

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        x              : (B, S, D)  – keys / values
        attn_mask      : (B*H, Q, S) bool or float,   True | -inf 👉 block
        key_padding_mask: (B, S) optional padding
        """
        B = x.size(0)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        # nn.MultiheadAttention automatically handles the head dimension
        attn_out, attn_w = self.mha(
            query=queries,
            key=x, value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        return attn_out, attn_w