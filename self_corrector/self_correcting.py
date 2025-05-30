import torch
import torch.nn as nn
import torch.optim as optim

class SelfCorrectingTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dim_feedforward: int,
                 dropout: float = 0.1, activation: str = "relu",
                 internal_lr: float = 1e-4, shift_amount: int = 1,
                 batch_first: bool = False):
        """
        A Transformer layer that performs a self-contained forward/backward iteration
        on each forward call, using shifted inputs as targets for an internal loss.

        Args:
            d_model: The number of expected features in the input (required).
            n_head: The number of heads in the multiheadattention models (required).
            dim_feedforward: The dimension of the feedforward network model (default=2048).
            dropout: The dropout value (default=0.1).
            activation: The activation function of the intermediate layer, can be a string
                        ("relu" or "gelu") or a unary callable (default="relu").
            internal_lr: Learning rate for the internal optimizer (default=1e-4).
            shift_amount: The amount to shift the input sequence to create the target
                          for the internal loss (default=1). Must be positive.
            batch_first: If True, then the input and output tensors are provided
                         as (batch, seq, feature). Default: False (seq, batch, feature).
        """
        super().__init__()

        if shift_amount <= 0:
            raise ValueError("shift_amount must be positive.")

        self.d_model = d_model
        self.n_head = n_head
        self.batch_first = batch_first
        self.shift_amount = shift_amount

        # The core transformer layer whose weights will be internally updated
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward, dropout, activation, batch_first=batch_first
        )

        # Internal optimizer for this layer's parameters only
        # This optimizer will update self.transformer_layer.parameters()
        self.internal_optimizer = optim.Adam(self.transformer_layer.parameters(), lr=internal_lr)

        # Loss function for the internal predictive task (e.g., predicting the next embedding)
        self.loss_fn = nn.MSELoss() # Using MSELoss for embedding prediction

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the self-correcting forward pass.

        Args:
            src: The input sequence to the layer.
                 Shape: (S, N, E) if batch_first=False, (N, S, E) if batch_first=True.
                 S=sequence length, N=batch size, E=embedding dimension (d_model).
            src_mask: The additive mask for the src sequence.
                      Shape: (S, S) or (N*nhead, S, S).
            src_key_padding_mask: The byte mask for padding tokens in the src sequence.
                                  Shape: (N, S).

        Returns:
            The output tensor from the transformer layer, after internal correction.
            Shape: Same as input `src`.
        """

        seq_dim = 0 if not self.batch_first else 1
        current_seq_len = src.size(seq_dim)

        # --- 1. First Forward Pass (for internal loss calculation) ---
        # This output is based on the current weights of self.transformer_layer.
        # It will be used to calculate the internal loss.
        # Gradients for `src` (if any from previous layers) will flow through this.

        internal_src = src.clone().detach()  # Detach to avoid gradients from previous layers affecting this pass
        out_for_loss = self.transformer_layer(internal_src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # --- 2. Check if internal update is possible (sequence length) ---
        if current_seq_len <= self.shift_amount:
            # Sequence is too short to create shifted targets for the internal loss.
            # Skip internal update and return the output from the first pass.
            # The layer behaves like a standard TransformerEncoderLayer in this case.
            return out_for_loss

        # --- 3. Prepare Predictions and Targets for Internal Loss ---
        # Predictions: Output of the layer, excluding the last `shift_amount` elements.
        # Targets: Input `src` shifted by `shift_amount`, excluding the first `shift_amount` elements from `src`.
        # We detach the targets to ensure they are treated as fixed labels and
        # gradients don't flow back into `src` through this target path during internal backprop.

        if not self.batch_first: # src/out_for_loss shape: (S, N, E)
            predictions = out_for_loss[:-self.shift_amount, :, :]
            targets = src[self.shift_amount:, :, :].detach()
        else: # src/out_for_loss shape: (N, S, E)
            predictions = out_for_loss[:, :-self.shift_amount, :]
            targets = src[:, self.shift_amount:, :].detach()

        # This check should ideally be covered by `current_seq_len <= self.shift_amount`,
        # but as a safeguard if slicing results in an empty tensor for some reason.
        if predictions.size(seq_dim) == 0:
            return out_for_loss

        # --- 4. Calculate Internal Loss ---
        internal_loss = self.loss_fn(predictions, targets)

        # --- 5. Perform Internal Backward Pass and Update Weights ---
        # Zero gradients for the internal optimizer
        self.internal_optimizer.zero_grad()
        # Calculate gradients for self.transformer_layer.parameters() w.r.t. internal_loss
        internal_loss.backward()
        # Update the weights of self.transformer_layer
        self.internal_optimizer.step()

        # --- 6. Second Forward Pass (with updated weights) ---
        # The parameters of self.transformer_layer have now been updated.
        # To ensure the output returned to the main computation graph (or the next layer)
        # reflects this internal correction, we perform another forward pass.
        # If `src` requires gradients, the graph from `src` to `final_out` will
        # correctly use the *updated* weights for any subsequent external backpropagation.
        final_out = self.transformer_layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return final_out

if __name__ == '__main__':
    # Configuration
    d_model = 16
    nhead = 4
    dim_feedforward = 32
    dropout_rate = 0.1
    learning_rate_internal = 0.01 # Relatively high LR for internal adaptation demonstration
    seq_len = 10
    batch_size = 2

    # --- Test Case 1: batch_first = False ---
    print("--- Test Case: batch_first = False ---")
    layer_bf_false = SelfCorrectingTransformerLayer(
        d_model, nhead, dim_feedforward, dropout_rate,
        internal_lr=learning_rate_internal, batch_first=False, shift_amount=1
    )
    # Input tensor: (S, N, E)
    test_input_bf_false = torch.randn(seq_len, batch_size, d_model)
    test_input_bf_false_clone = test_input_bf_false.clone().requires_grad_(True)


    # Check initial parameter value (e.g., sum of a weight matrix)
    initial_param_sum_bf_false = layer_bf_false.transformer_layer.self_attn.in_proj_weight.data.sum().item()
    print(f"Initial in_proj_weight sum: {initial_param_sum_bf_false:.4f}")

    # Forward pass
    output_bf_false = layer_bf_false(test_input_bf_false_clone)

    # Check if parameter value has changed
    updated_param_sum_bf_false = layer_bf_false.transformer_layer.self_attn.in_proj_weight.data.sum().item()
    print(f"Updated in_proj_weight sum after one forward: {updated_param_sum_bf_false:.4f}")
    assert initial_param_sum_bf_false != updated_param_sum_bf_false, "Parameter should have been updated internally."
    print(f"Output shape (S, N, E): {output_bf_false.shape}")
    assert output_bf_false.shape == (seq_len, batch_size, d_model)

    # Simulate external loss and backpropagation
    external_loss_bf_false = output_bf_false.mean()
    external_loss_bf_false.backward()
    print(f"Gradient on in_proj_weight after external loss: "
          f"{layer_bf_false.transformer_layer.self_attn.in_proj_weight.grad is not None}")
    assert layer_bf_false.transformer_layer.self_attn.in_proj_weight.grad is not None, "Grad should exist from external loss."
    print(f"Gradient on input tensor after external loss: {test_input_bf_false_clone.grad is not None}")
    assert test_input_bf_false_clone.grad is not None, "Grad should flow back to input."
    print("-" * 30)

    # --- Test Case 2: batch_first = True ---
    print("\n--- Test Case: batch_first = True ---")
    layer_bf_true = SelfCorrectingTransformerLayer(
        d_model, nhead, dim_feedforward, dropout_rate,
        internal_lr=learning_rate_internal, batch_first=True, shift_amount=1
    )
    # Input tensor: (N, S, E)
    test_input_bf_true = torch.randn(batch_size, seq_len, d_model)
    test_input_bf_true_clone = test_input_bf_true.clone().requires_grad_(True)

    initial_param_sum_bf_true = layer_bf_true.transformer_layer.self_attn.in_proj_weight.data.sum().item()
    print(f"Initial in_proj_weight sum: {initial_param_sum_bf_true:.4f}")

    output_bf_true = layer_bf_true(test_input_bf_true_clone)

    updated_param_sum_bf_true = layer_bf_true.transformer_layer.self_attn.in_proj_weight.data.sum().item()
    print(f"Updated in_proj_weight sum after one forward: {updated_param_sum_bf_true:.4f}")
    assert initial_param_sum_bf_true != updated_param_sum_bf_true, "Parameter should have been updated internally."
    print(f"Output shape (N, S, E): {output_bf_true.shape}")
    assert output_bf_true.shape == (batch_size, seq_len, d_model)

    external_loss_bf_true = output_bf_true.mean()
    external_loss_bf_true.backward()
    print(f"Gradient on in_proj_weight after external loss: "
          f"{layer_bf_true.transformer_layer.self_attn.in_proj_weight.grad is not None}")
    assert layer_bf_true.transformer_layer.self_attn.in_proj_weight.grad is not None, "Grad should exist from external loss."
    print(f"Gradient on input tensor after external loss: {test_input_bf_true_clone.grad is not None}")
    assert test_input_bf_true_clone.grad is not None, "Grad should flow back to input."
    print("-" * 30)

    # --- Test Case 3: Sequence too short for internal update ---
    print("\n--- Test Case: Sequence too short ---")
    short_seq_len = 1 # shift_amount is 1 by default
    layer_short_seq = SelfCorrectingTransformerLayer(
        d_model, nhead, dim_feedforward, dropout_rate,
        internal_lr=learning_rate_internal, batch_first=False, shift_amount=1 # shift_amount = 1
    )
    # Input tensor: (S, N, E) where S=1
    test_input_short = torch.randn(short_seq_len, batch_size, d_model)

    initial_param_sum_short = layer_short_seq.transformer_layer.self_attn.in_proj_weight.data.sum().item()
    print(f"Initial in_proj_weight sum: {initial_param_sum_short:.4f}")

    output_short = layer_short_seq(test_input_short)

    updated_param_sum_short = layer_short_seq.transformer_layer.self_attn.in_proj_weight.data.sum().item()
    print(f"Updated in_proj_weight sum (should be same): {updated_param_sum_short:.4f}")
    assert initial_param_sum_short == updated_param_sum_short, "Parameter should NOT have been updated for short sequence."
    print(f"Output shape (S, N, E): {output_short.shape}")
    assert output_short.shape == (short_seq_len, batch_size, d_model)
    print("-" * 30)

    print("\nAll tests completed.")