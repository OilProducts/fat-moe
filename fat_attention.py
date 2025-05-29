import torch
import torch.nn as nn
from typing import Optional, Literal

class AutoregressiveQueryAugmentation(nn.Module):
    """
    Augments token hidden states in an autoregressive LLM by incorporating
    features derived from learned queries attending to those token states.

    This module is designed to fit within a Transformer decoder layer,
    typically after the main self-attention block and before the FFN,
    or as part of the residual connection around the FFN.
    """
    def __init__(self,
                 embed_dim: int,
                 num_learned_queries: int,
                 num_attention_heads: int,
                 dropout: float = 0.1,
                 fusion_method: Literal['add', 'concat'] = 'add',
                 device: Optional[torch.device | str] = None,
                 dtype: Optional[torch.dtype] = None): # Added dtype
        """
        Initializes the AutoregressiveQueryAugmentation module.

        Args:
            embed_dim: The embedding dimension of the token hidden states and queries.
            num_learned_queries: The number of learned query vectors.
            num_attention_heads: Number of attention heads for the MHA that uses
                                 the learned queries.
            dropout: Dropout probability for the attention mechanism and fusion.
            fusion_method: Method to fuse learned query features with token states.
                           Options: 'add' or 'concat'.
            device: The device ('cpu', 'cuda', etc.) for parameters.
            dtype: The data type for parameters (e.g., torch.float32).
        """
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        if embed_dim % num_attention_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        if fusion_method not in ['add', 'concat']:
            raise ValueError("fusion_method must be 'add' or 'concat'")

        self.embed_dim = embed_dim
        self.num_learned_queries = num_learned_queries
        self.fusion_method = fusion_method

        # Learned query parameters
        self.learned_queries = nn.Parameter(torch.empty(num_learned_queries, embed_dim, **factory_kwargs))
        # Initialize learned queries (can use Xavier/Kaiming or simple Gaussian)
        nn.init.normal_(self.learned_queries, std=0.02)

        # Multi-Head Attention layer for learned queries to attend to token states
        self.query_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
            **factory_kwargs
        )

        # LayerNorm for the output of the learned query attention path before aggregation
        self.norm_learned_output = nn.LayerNorm(embed_dim, **factory_kwargs)

        if self.fusion_method == 'concat':
            self.fusion_linear = nn.Linear(embed_dim * 2, embed_dim, **factory_kwargs)

        # LayerNorm for the final output of this augmentation module
        self.norm_final_output = nn.LayerNorm(embed_dim, **factory_kwargs)

        # Dropout on the aggregated features before fusion
        self.fusion_dropout = nn.Dropout(dropout)

    def forward(self,
                token_hidden_states: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of the augmentation.

        Args:
            token_hidden_states: Tensor of shape (batch_size, seq_len, embed_dim).
                                 These are typically the outputs from a previous
                                 self-attention block in an LLM decoder layer.
            key_padding_mask: Optional tensor of shape (batch_size, seq_len)
                              indicating padded tokens in token_hidden_states
                              when they act as keys/values for the query_attention.
                              True/1 indicates a padded token that should be ignored.

        Returns:
            augmented_hidden_states: Tensor of shape (batch_size, seq_len, embed_dim)
                                     after augmentation.
        """
        batch_size, seq_len, _ = token_hidden_states.shape

        # Expand learned queries for the batch
        # Target shape for query_attention: (batch_size, num_learned_queries, embed_dim)
        # self.learned_queries shape: (num_learned_queries, embed_dim)
        queries_for_batch = self.learned_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Learned queries attend to the token_hidden_states
        # Q: queries_for_batch (batch_size, num_learned_queries, embed_dim)
        # K, V: token_hidden_states (batch_size, seq_len, embed_dim)
        # Output shape: (batch_size, num_learned_queries, embed_dim)
        learned_query_attended_features, _ = self.query_attention(
            query=queries_for_batch,
            key=token_hidden_states,
            value=token_hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False # Typically False unless you need to inspect weights
        )

        # Normalize the output from the learned query attention path
        # Input shape: (batch_size, num_learned_queries, embed_dim)
        learned_query_attended_features = self.norm_learned_output(learned_query_attended_features)

        # Aggregate these features across the 'num_learned_queries' dimension.
        # A simple mean is a common choice to get a single summary vector per batch item.
        # Shape: (batch_size, embed_dim)
        aggregated_learned_features = learned_query_attended_features.mean(dim=1)

        # Apply dropout to aggregated features
        aggregated_learned_features = self.fusion_dropout(aggregated_learned_features)

        # Expand aggregated features to match sequence length for element-wise fusion
        # Shape: (batch_size, seq_len, embed_dim)
        aggregated_learned_features_expanded = aggregated_learned_features.unsqueeze(1).expand(-1, seq_len, -1)

        # Fuse with original token_hidden_states
        # Note: The original token_hidden_states are often part of a residual connection path
        # in a full Transformer layer. This module adds to that state.
        if self.fusion_method == 'add':
            # This acts like a parallel branch adding information
            augmented_hidden_states = token_hidden_states + aggregated_learned_features_expanded
        elif self.fusion_method == 'concat':
            concatenated = torch.cat((token_hidden_states, aggregated_learned_features_expanded), dim=2)
            augmented_hidden_states = self.fusion_linear(concatenated)
        else:
            # This case should be caught by the __init__ check
            raise InternalError("Invalid fusion_method. This should not be reached.")


        # Final LayerNorm for the output of this augmentation step
        augmented_hidden_states = self.norm_final_output(augmented_hidden_states)

        return augmented_hidden_states


class StandardDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # --- POINT OF INTEGRATION for AutoregressiveQueryAugmentation ---
        self.query_augment = AutoregressiveQueryAugmentation(
            embed_dim=embed_dim,
            num_learned_queries=16, # Example: 16 learned queries
            num_attention_heads=num_heads, # Can be same or different from main self_attn
            dropout=dropout,
            fusion_method='add'
        )
        # self.norm_augment = nn.LayerNorm(embed_dim) # Already in AutoregressiveQueryAugmentation
        # self.dropout_augment = nn.Dropout(dropout) # Already in AutoregressiveQueryAugmentation

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(), # or GELU
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None): # memory and memory_key_padding_mask for cross-attn

        # 1. Masked Self-Attention
        sa_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)
        x = tgt + self.dropout1(sa_out) # Add & Norm (residual connection)
        x = self.norm1(x)

        # --- INTEGRATION ---
        # x is now the token_hidden_states after self-attention and first add&norm
        # Pass the padding mask relevant for x if it's used as K,V
        augmented_x = self.query_augment(x, key_padding_mask=tgt_key_padding_mask)
        # The AutoregressiveQueryAugmentation module has its own internal residual logic (if 'add')
        # and final layernorm.
        # If query_augment's 'add' fusion is seen as a parallel branch to x,
        # then x for FFN could be `x + augmented_x_contribution` or just `augmented_x`.
        # Given the internal structure (token_hidden_states + aggregated_learned_features_expanded),
        # `augmented_x` already incorporates `x`.
        x_for_ffn = augmented_x

        # 2. Feed-Forward Network
        ffn_out = self.ffn(x_for_ffn)
        x = x_for_ffn + self.dropout2(ffn_out) # Add & Norm
        x = self.norm2(x)

        return x