#!/usr/bin/python3
"Implementation of Encoder for a Transformer model"

from typing import Optional

import torch
from torch import nn
from day3_mha import MultiHeadAttention

class PositionWiseFeedForward(nn.Module):
    "The feed-forward neural network implementation for a Transformer model"
    def __init__(self, d_model: int, d_ff: int):
        """Constructor for this class

        Params:
         d_model: The embedding dimension of the Transformer model
         d_ff: The number of neurons in the hidden layer of the feed-forward network
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Forward propagation for the given input tensor x
        Shape of x: [num_batches, seq_len, d_model]
        """
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    "The encoder layer for a Transformer model"
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.10):
        """Constructor for this class

        Params:
         d_model: The embedding dimension of the Transformer model
         num_heads: Number of heads in the Attention mechanism
         d_ff: Number of neurons in the hidden layer of the feed-forward network
         dropout: The drop-out rate for the neurons in the feed-forward network
        """
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)

        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward propagation for the given input tensor x
        Shape of x: [num_batches, seq_len, d_model]
        """
        # 1. Multi-head attention + Residual + norm
        attn_output = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Feed-forward + Residual + norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


def test():
    "Test function of this script"
    pass


if __name__ == "__main__":
    # Call the test function
    test()
