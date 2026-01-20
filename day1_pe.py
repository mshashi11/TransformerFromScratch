#!/usr/bin/python3
"Implementation of Positional Encoding for a Transformer"

import math
import torch
from torch import nn

import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    "Class for defining positional encoding for a Transformer"
    def __init__(self, d_model: int, max_len: int = 5000):
        """Constructor for this class. It creates a static encoding for each position in
        the sequence, and each dimension of the embedding, which can then be added to the
        embedding representation of tokens in a sequence.
        """
        super().__init__()

        # Initial matrix: all zeroes
        pe = torch.zeros(max_len, d_model)

        # Positional tensor with values ranging from 0 to max_len - 1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Division term: 10000 ^ (2 * pos / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Applying sine to even indices (0, 2, 4, ...) of dimension
        pe[:, 0::2] = torch.sin(position * div_term)

        # Applying cosine to odd indices (1, 3, 5, ...) of dimension
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)

        # Register it as a buffer: not a learnable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """Forward propagation for a given input x. It adds the positional encoding to each
        entry of the embedding in the given tensor x"""
        # x is a tensor of dimension [batch, sequence_length, d_model]
        # Add the positional encodings to x
        # Note that here the assumption is that the length of x is less than the
        # max_len assumed in the constructor of this class
        return x + self.pe[:, :x.size(1), :]


def test():
    "Test function for testing the classes in this script"
    # 1. Configuration
    vocab_size = 20 # Number of words in the vocabulary
    d_model = 32 # Each word will be represented by 32 numbers in the embedding
    seq_len = 10 # 10 words in the sentence

    # 2. Create dummy word ids
    # Shape: [Batch=1, Seq_Len=10]
    t = torch.randint(0, vocab_size, (1, seq_len))

    # 3. Convert the ids to embeddings
    embedding_layer = nn.Embedding(vocab_size, d_model)
    embedded_t = embedding_layer(t) # New shape: [1, 10, 32]

    # 4. Add positional information
    pe = PositionalEncoding(d_model, max_len=seq_len)
    output = pe(embedded_t)

    # This should be [1, 10], the dimension we specified when creating t
    print(f"Input shape (IDs): {t.shape}")

    # This should be [1, 10, 32]: Each word has been converted into
    # its embedding of dimension 32
    print(f"After Embedding: {embedded_t.shape}")

    # This should also be [1, 10, 32], since we are only adding to the
    # embedding values using positional encodings
    print(f"Final Output shape: {output.shape}")


def visualize_pe():
    "Test function to visualize and verify the Positional Encoding"
    d_model = 512
    max_len = 100
    pe_layer = PositionalEncoding(d_model, max_len)

    # Extract the matrix from the buffer
    # Shape: [100, 512]
    matrix = pe_layer.pe.squeeze(0).cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(matrix, cmap='RdBu')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Token Position')
    plt.colorbar(label='Signal Strength')
    plt.title("Positional Encoding Matrix (Sine/Cosine Waves)")
    plt.savefig("pos_encoding.png")
    print("Visualization saved as pos_encoding.png")


if __name__ == "__main__":
    # Call the test function
    test()
    visualize_pe()
