#!/usr/bin/python3
"""Implementation of multi-headed attention in a Transformer architecture"""

from typing import Optional

import torch
from torch import nn
from day2_attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    "Class for implementing mult-head attention mechanism in a Transformer model"
    def __init__(self, d_model: int, num_heads: int):
        """Constructor of this class

        Params:
         d_model: The embedding dimension of the transformer
         num_heads: The number of heads in the attention mechanism
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Ensure that the model dimension is divisible by the number of heads
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads

        # Learnable weight matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model) # Output projection

    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Split the heads for attention mechanism

        Params:
         x: The input tensor, shape [num_batches, seq_len, d_model]
         batch_size: The size of each batch

        Output:
         The resulting tensor, with shape [num_batches, num_heads, seq_len, d_k]
         This matches the shape of the input for the function we have implemented for
         attention function previously

        """
        # Shape of x: [num_batches, seq_len, d_model]
        # Reshape to: [num_batches, seq_len, num_heads, d_k]
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        # Transpose to: [num_batches, num_heads, seq_len, d_k]
        return x.transpose(1, 2)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward propagation for this multi-head attention mechanism"""
        batch_size = q.size(0)

        # 1. Linear projections [num_batches, seq_len, d_model]
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        # 2. Split into heads [num_batches, seq_len, num_heads, d_k]
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 3. Apply scaled dot-product attention
        output, _ = scaled_dot_product_attention(q, k, v, mask)

        # 4. Concatenate the heads back together
        # Move num_heads back: [Batch, Seq_Len, Num_Heads, d_k]
        output = output.transpose(1, 2).contiguous()
        # Flatten the last two dimensions: [Batch, Seq_Len, d_model]
        output = output.view(batch_size, -1, self.d_model)

        # 5. Final linear projection
        return self.W_o(output)


def test():
    "Test function of this script"
    num_batches = 1
    seq_len = 4
    d_model = 32
    num_heads = 8

    input_tensor = torch.normal(mean=0.0, std=1.0, size=(num_batches, seq_len, d_model))

    mha = MultiHeadAttention(d_model, num_heads)
    output_tensor = mha(input_tensor, input_tensor, input_tensor)

    # Input and output shape should match exactly
    print(input_tensor.shape)
    print(output_tensor.shape)

    print("\nInput tensor:")
    print(input_tensor)

    print("\nOutput tensor:")
    print(output_tensor)


if __name__ == "__main__":
    # Call the test function
    test()
