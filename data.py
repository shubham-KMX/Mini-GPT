"""
Parsing the Play: Data and Tokenization
=======================================

This module implements the data pipeline for a character-level, decoder-only
Transformer trained on the Tiny Shakespeare dataset:

    1. Fetch and read the raw text.
    2. Build a character-level vocabulary.
    3. Encode / decode between strings and integer IDs.
    4. Split into train / validation tensors.
    5. Sample batches of (input, target) sequences for autoregressive training.

Run this file directly to see the pipeline in action:

    python data.py
"""

from __future__ import annotations

import os
import urllib.request

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_PATH = os.path.join(os.path.dirname(__file__), "input.txt")

BATCH_SIZE = 4   # how many independent sequences we process in parallel
BLOCK_SIZE = 8   # maximum context length used to predict the next token
TRAIN_SPLIT = 0.9  # fraction of data used for training (rest is validation)


# ---------------------------------------------------------------------------
# 1. Fetching and inspecting the dataset
# ---------------------------------------------------------------------------
def load_text(path: str = DATA_PATH, url: str = DATA_URL) -> str:
    """Return the raw Tiny Shakespeare text, downloading it once if needed."""
    if not os.path.exists(path):
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 2 & 3. Character-level tokenizer (encode / decode)
# ---------------------------------------------------------------------------
class CharTokenizer:
    """A minimal character-level tokenizer.

    The vocabulary is every unique character in the source text, sorted so the
    mapping is deterministic. `encode` turns a string into a list of integer
    IDs; `decode` turns a list of IDs back into a string.
    """

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """String -> list of integer token IDs."""
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        """List of integer token IDs -> string."""
        return "".join(self.itos[i] for i in ids)


# ---------------------------------------------------------------------------
# 4. Train / validation split
# ---------------------------------------------------------------------------
def build_dataset(text: str, tokenizer: CharTokenizer, train_split: float = TRAIN_SPLIT):
    """Encode the full text and split it into train / validation tensors."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return data, train_data, val_data


# ---------------------------------------------------------------------------
# 5. Batch sampling
# ---------------------------------------------------------------------------
def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    block_size: int = BLOCK_SIZE,
):
    """Sample a random batch of input (x) and target (y) sequences.

    For each sampled starting index `i`, the input is `data[i : i+block_size]`
    and the target is the same window shifted forward by one position,
    `data[i+1 : i+block_size+1]`. This shift is what makes the task
    autoregressive: at every position, predict the *next* token.
    """
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i : i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# Demo / self-check
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(1337)  # reproducibility

    # 1. Fetch and inspect
    text = load_text()
    print("Length of dataset in characters:", len(text))
    print("--- Sample ---")
    print(text[:150])

    # 2. Vocabulary
    tokenizer = CharTokenizer(text)
    print("\nVocabulary:", "".join(tokenizer.chars))
    print("Vocabulary size:", tokenizer.vocab_size)

    # 3. Encode / decode round-trip
    print("\nEncoded:", tokenizer.encode("hello there"))
    print("Decoded:", tokenizer.decode(tokenizer.encode("hello there")))

    # 4. Tensor + split
    data, train_data, val_data = build_dataset(text, tokenizer)
    print("\n", data.shape, data.dtype)
    print("First 20 tokens:", data[:20])
    print(f"Train tokens: {len(train_data)}  |  Val tokens: {len(val_data)}")

    # 5. Sample a batch
    xb, yb = get_batch("train", train_data, val_data)
    print("\nInputs (x):")
    print(xb.shape)
    print(xb)
    print("\nTargets (y):")
    print(yb.shape)
    print(yb)

    # Show the hidden context-target pairs inside the batch
    print("\n--- Context -> Target pairs in the first sequence ---")
    for t in range(BLOCK_SIZE):
        context = xb[0, : t + 1].tolist()
        target = yb[0, t].item()
        print(f"context {context} -> target {target}")


if __name__ == "__main__":
    main()
