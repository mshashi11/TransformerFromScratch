#!/usr/bin/python3
"Implementation of scaled dot product attention for transformers"

import math
from typing import Optional

import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function for scaled dot production attention calculation in a Transformer model.

    Parameters:
     q: Query tensor, shape [batch_size, num_heads, seq_len, key_dim]
     k: Key tensor, shape [batch_size, num_heads, seq_len, key_dim]
     v: Value tensor, shape [batch_size, num_heads, seq_len, key_dim]
     mask: Mask tensor to apply for given attention head

    Return:
     scaled dot production attention, given by the following formula:

     softmax(Q.K^T/sqrt(d_k)).V
    """
    # Get the dimension of the key vectors
    d_k = q.size(-1)

    # 1. Calculate scores
    # scores has shape [batch_size, num_heads, seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. Apply mask, if given
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Apply softmax function to scores
    # attention_weights has shape [batch_size, num_heads, seq_len, seq_len]
    # Along the last dimension, the entries of the tensor sum up to 1.0
    attention_weights = F.softmax(scores, dim=-1)

    # 4. Multiply by values
    # output has shape [batch_size, num_heads, seq_len, head_dim]
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """Create a 2D tensor of shape [size, size] corresponding to look-ahead
    mask for next-token generation in a decoder"""
    mask = torch.zeros(size, size)

    for i in range(size):
        mask[i, :i+1] = 1

    return mask


def test():
    "Test function for this script"
    num_batches = 1
    num_heads = 1
    seq_len = 4
    dim = 32

    # Create random Q, K, V tensors of shape:
    # [num_batches, num_heads, seq_len, dim]
    q = torch.normal(mean=0.0, std=5.0, size=(num_batches, num_heads, seq_len, dim))
    k = torch.normal(mean=0.0, std=5.0, size=(num_batches, num_heads, seq_len, dim))
    v = torch.normal(mean=0.0, std=5.0, size=(num_batches, num_heads, seq_len, dim))

    # Create a mask of shape [seq_len, seq_len]
    mask = create_look_ahead_mask(seq_len)

    # Verify the shape of each tensor: [num_batches, num_heads, seq_len, dim]
    print(f"Query tensor shape: {q.shape}")
    print(f"Key tensor shape: {k.shape}")
    print(f"Value tensor shape: {v.shape}")

    # Verify the shape of mask: [seq_len, seq_len]
    print(f"Mask tensor shape: {mask.shape}")

    # Verify the mask tensor
    print("\nMask tensor:")
    print(mask)

    output, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    # Verify the attention_weights tensor shape
    # it should be [num_batches, num_heads, seq_len, seq_len]
    print(f"Attention weights shape: {attention_weights.shape}")

    # Verify the attention weights tensor
    # Each row in this tensor should sum up to 1.0
    print("\nAttention weights:")
    print(attention_weights)

    # Verify the output tensor shape
    # it should be [num_batches, num_heads, seq_length, dim]
    print(f"\nOutput shape: {output.shape}")

    # Check the output tensor
    print("\nOutput tensor:")
    print(output)


if __name__ == "__main__":
    # Call the test function
    test()
