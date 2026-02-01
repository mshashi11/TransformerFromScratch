#!/usr/bin/python3
"Implementation of a full Transformer model"

import os
import json
from typing import Tuple

import torch
import requests
from torch import nn
from torch.utils.data import Dataset, DataLoader

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


class TransformerTrainingDataset(Dataset):
    "Class for providing the training dataset for the Transformer class"
    def __init__(self, data: torch.Tensor, seq_length: int):
        "Constructor for this class"
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        "Override function from the super class"
        # We can create a sequence starting at almost any index
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        # Grab a chunk of text of length seq_length + 1
        chunk = self.data[idx : idx + self.seq_length + 1]

        # inp: The sequence the Encoder sees
        # tar_inp: The sequence the Decoder sees (Teacher Forcing)
        # Both are the same in this character-generation setup
        inp = chunk[:-1]
        tar_inp = chunk[:-1]

        # tar_real: The actual 'next characters' we want to predict
        # This is shifted one to the right
        tar_real = chunk[1:]

        return inp, tar_inp, tar_real


def download_text(filename: str, url: str) -> None:
    "Download the file from the given URL, if not present in local disk"
    if not os.path.exists(filename):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(response.text)
        print("Download complete.")


def character_level_tokenizer(text: str) -> Tuple[torch.Tensor, dict[str, int]]:
    "Convert the given text into a set of character level tokens"
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}

    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of characters in text: {len(text)}")

    # Encode the entire text
    data = torch.tensor([char_to_int[c] for c in text], dtype=torch.long)
    return data, char_to_int


def create_masks(inp: torch.Tensor, tar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    "Genereate the masks needed for the encoder/decoder layers of the transfomer"
    # 1. Encoder Padding Mask (For the source sequence)
    # Marks where the input has padding (0s)
    enc_mask = None # (inp != 0).unsqueeze(1).unsqueeze(2)

    # 2. Decoder Padding Mask (Used in Cross-Attention)
    # Tells decoder to ignore padding in the encoder output
    dec_mask = None # (inp != 0).unsqueeze(1).unsqueeze(2)

    # 3. Look-Ahead Mask (Day 2 implementation)
    # Prevents decoder from seeing the future
    size = tar.size(1)
    look_ahead_mask = torch.tril(torch.ones(size, size)).to(tar.device)

    return enc_mask, look_ahead_mask, dec_mask


def train_transformer(
        model: Transformer,
        train_loader: DataLoader,
        device: torch.device,
        vocab_size: int,
        learning_rate: float = 0.0001,
        max_iter: int = 1,
        target_loss: float = 1.8
) -> None:
    "Train the Transformer model with given training data"
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Model training loop
    for epoch in range(max_iter):
        epoch_loss = 0
        batch_num = 0
        for inp, tar_inp, tar_real in train_loader:
            # Move data to the correct device
            inp, tar_inp, tar_real = inp.to(device), tar_inp.to(device), tar_real.to(device)

            # Forward pass
            optimizer.zero_grad()
            enc_mask, look_ahead_mask, dec_mask = create_masks(inp, tar_inp)
            output = model(inp, tar_inp, enc_mask, look_ahead_mask, dec_mask)
            loss = criterion(output.view(-1, vocab_size), tar_real.view(-1))
            if loss <= target_loss:
                print(f"Target loss of {target_loss:.2f} attained, exiting training")
                return

            epoch_loss += loss.item()

            if batch_num % 100 == 0:
                print(f"Epoch: {epoch:3d} | Batch: {batch_num:4d} | Loss: {loss:.2f} | Epoch Loss: {epoch_loss:.2f}")

            # Backward pass
            loss.backward()
            optimizer.step()
            batch_num += 1

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {epoch_loss:.2f}")


def save_vocab(char_to_int: dict[str, int]) -> None:
    "Save the vocabulary to json file to ensure consistency between training and inference"
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump({"char_to_int": char_to_int}, f)


def test():
    "Test function for this script"
    print("Testing the Transformer class")
    filename = "tiny_shakespeare.txt"
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # Download the text, if not present locally
    download_text(filename, url)
    text = ""
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    encoded_text, char_to_int = character_level_tokenizer(text)
    save_vocab(char_to_int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # How many characters the model is looking at once
    seq_length = 128
    dataset = TransformerTrainingDataset(encoded_text, seq_length)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Initialize the transformer
    transformer = Transformer(
        num_layers=6,
        d_model=64,
        num_heads=8,
        d_ff=256,
        input_vocab_size=len(char_to_int),
        target_vocab_size=len(char_to_int),
        pe_max_len=seq_length,
        dropout=0.10
    ).to(device)

    # Train the transformer with the given dataset
    train_transformer(transformer, dataloader, device, len(char_to_int), max_iter=10, target_loss=1.2)

    # Save the trained model to the local directory
    torch.save(transformer.state_dict(), "shakespeare_llm.pth")
    print("Model weights saved to shakespeare_llm.pth")


if __name__ == "__main__":
    # Call the test function
    test()
