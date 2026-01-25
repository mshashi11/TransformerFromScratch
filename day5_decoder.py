#!/usr/bin/python3
"Implementation of the Decoder in a Transformer architecture"

from typing import Optional

import torch
from torch import nn
import day2_attention as attention
from day3_mha import MultiHeadAttention
from day4_encoder import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    "Defining the class for the Decoder layer"
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.10):
        """
        Constructor of this class

        Params:
         d_model: Embedding dimension of the model
         num_heads: Number of heads in the Attention mechanism
         d_ff: Number of neurons in the hidden layer of the feed-forward network
         dropout: The drop-out rate for the neurons in the feed-forward network
        """
        super().__init__()

        # Masked self-attention
        self.mha1 = MultiHeadAttention(d_model, num_heads)

        # Cross-attention (Encoder-Decoder)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            enc_output: torch.Tensor,
            look_ahead_mask: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward propagation for the Decoder

        Params:
         x: The sequence generated so far, shape [num_batches, seq_len, d_model]
         enc_output: Output tensor generated from the Encoder, shape [num_batches, seq_len, d_model]
         look_ahead_mask: Look ahead mask, shape [seq_len, seq_len]
         padding_mask: Padding mask, shape [seq_len, seq_len]

        Returns: Decoder output of shape [num_batches, seq_len, d_model]
        """
        # Step 1: Masked Self-Attention
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.dropout(attn1))

        # Step 2: Cross-Attention
        # Q comes from Decoder (x), K and V come from Encoder (enc_output)
        attn2 = self.mha2(x, enc_output, enc_output, padding_mask)
        x = self.norm2(x + self.dropout(attn2))

        # Step 3: Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


def test():
    "The test function for this script"
    num_batches = 1
    seq_len = 4
    d_model = 32

    num_heads = 8
    d_ff = 64
    dropout = 0.10

    # Creating a dummy encoder output
    enc_output = torch.normal(mean=0.0, std=1.0, size=(num_batches, seq_len, d_model))

    # Creating a dummy decoder input
    dec_input = torch.normal(mean=0.0, std=1.0, size=(num_batches, seq_len, d_model))

    # Look-ahead mask for a sequence of length 4
    look_ahead_mask = attention.create_look_ahead_mask(seq_len)

    # Create the decoder object
    decoder = DecoderLayer(d_model, num_heads, d_ff, dropout)

    dec_output = decoder.forward(dec_input, enc_output, look_ahead_mask)

    # This should print [num_matches, seq_len, d_model]
    print(dec_output.shape)

    print(dec_input)
    print(dec_output)


if __name__ == "__main__":
    # Call the test function here
    test()
