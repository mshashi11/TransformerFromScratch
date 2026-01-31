#!/usr/bin/python3
"Script for running inference on a trained Transformer model"

import torch
import torch.nn.functional as F
from day6_transformer import Transformer, character_level_tokenizer, create_masks

def load_model(checkpoint_path: str, vocab_size: int, device: torch.device) -> Transformer:
    "Load the Transformer model from the given checkpoint path"
    model = Transformer(
        num_layers=6,
        d_model=64,
        num_heads=8,
        d_ff=256,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        pe_max_len=128
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    # Set the model to evaluation mode only
    model.eval()
    return model


def generate_text(
        model: Transformer,
        start_str: str,
        char_to_int: dict[str, int],
        int_to_char: dict[int, str],
        device: torch.device,
        max_len: int = 500,
        temperature: float = 1.0
) -> str:
    "Run the inferenence using the given model, start string, and vocabulary"
    # 1. Convert string to tokens
    input_ids = [char_to_int[c] for c in start_str]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_str

    with torch.no_grad():
        for _ in range(max_len):
            # 1. Generate masks for the current sequence
            enc_mask, look_ahead_mask, dec_mask = create_masks(input_tensor, input_tensor)

            # 2. Forward pass
            # We feed the sequence to BOTH encoder and decoder to match training
            output = model(input_tensor, input_tensor, enc_mask, look_ahead_mask, dec_mask)

            # 3. Select the next token
            # Shape: [1, seq_len, vocab_size] -> [vocab_size]
            next_token_logits = output[0, -1, :] / temperature

            # Apply softmax and sample (Weighted Random choice)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            # 4. Append and check limits
            generated_text += int_to_char[next_token_id]
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

            # Safety: Ensure that we don't exceed the positional coding limit
            if input_tensor.size(1) >= 128:
                # Slide the window
                input_tensor = input_tensor[:, 1:]

    return generated_text


def test():
    "Test function for this script"
    # Load the necessary data for vocabulary etc here
    with open("tiny_shakespeare.txt", "r") as f:
        text = f.read()

    _, char_to_int, int_to_char = character_level_tokenizer(text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("shakespeare_llm.pth", len(char_to_int), device)

    print("\n--- GENERATED SHAKESPEARE ---")
    prompt = "OTHELLO:"
    print(generate_text(model, prompt, char_to_int, int_to_char, device, temperature=1.2))


if __name__ == "__main__":
    # Call the test function
    test()
