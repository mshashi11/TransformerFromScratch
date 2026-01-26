#!/usr/bin/python3
"Implementation of a full Transformer model"

import os
import requests

import torch
from torch import nn

from day1_pe import PositionalEncoding
from day4_encoder import EncoderLayer
from day5_decoder import DecoderLayer

class Transformer(nn.Module):
    "Full implementation of the Transformer model using Encoder/Decoder Layers"
    def __init__(
            self,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            input_vocab_size: int,
            target_vocab_size: int,
            pe_max_len: int,
            dropout: float = 0.10
    ):
        "Constructor for this class"
        super().__init__()

        # 1. Embeddings and Positional Encodings
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, pe_max_len)

        # 2. Stacked Encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )

        # 3. Final output head
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(
            self,
            inp: torch.Tensor,
            target: torch.Tensor,
            enc_mask: torch.Tensor,
            look_ahead_mask: torch.Tensor,
            dec_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward propagation for the given input

        Params:
         inp: Input tensor to the decoder, shape [batch_size, inp_seq_len]
         target: Target vocabulary set for the decoder, shape [batch_size, tar_seq_len]
         enc_mask: Encoder mask, shape [batch_size, 1, 1, inp_seq_len]
         look_ahead_mask: Look-ahead mask for the decoder, shape [tar_seq_len, tar_seq_len]
         dec_mask: decoder mask, shape [batch_size, 1, 1, inp_seq_len]

        Output: The next token generated from the Transformer,
                shape[batch_size, tar_seq_len, target_vocab_size]

        """
        # 1. Prepare the inputs for the Encoder
        enc_output = self.encoder_embedding(inp)
        enc_output = self.pos_encoding(enc_output)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, enc_mask)

        # 2. Prepare the inputs for the Decoder
        dec_output = self.decoder_embedding(target)
        dec_output = self.pos_encoding(dec_output)
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, look_ahead_mask, dec_mask)

        # 3. Project to vocabulary
        return self.final_layer(dec_output)


def download_text(filename: str, url: str) -> None:
    "Download the file from the given URL, if not present in local disk"
    if not os.path.exists(filename):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(filename, 'w') as f:
            f.write(response.text)
        print("Download complete.")


def character_level_tokenizer(text: str) -> torch.Tensor:
    "Convert the given text into a set of character level tokens"
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of characters in text: {len(text)}")
    
    # Encode the entire text
    data = torch.tensor([char_to_int[c] for c in text], dtype=torch.long)
    return data


def test():
    "Test function for this script"
    print("Testing the Transformer class")
    filename = "tiny_shakespeare.txt"
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # Download the text, if not present locally
    download_text(filename, url)
    text = ""
    with open(filename, "r") as f:
        text = f.read()

    character_level_tokenizer(text)


if __name__ == "__main__":
    # Call the test function
    test()
