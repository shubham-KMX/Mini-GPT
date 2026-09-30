"""
Context and Coordinates: The Embedding Layer
============================================

Neural networks cannot work with raw integer token IDs directly -- to a network
the integer 42 would look like "twice" 21, which is meaningless when the numbers
are just arbitrary character IDs. This module turns discrete token IDs into a
rich, continuous representation by combining two learned embedding tables:

    token_emb : "What is this token?"  (semantic identity)
    pos_emb   : "Where is this token?" (position in the sequence)

The two are added element-wise so every position carries both signals in the
same vector space. Because Transformers process a sequence in parallel, they
have no built-in sense of order; the positional embedding is what injects it.

Run this file directly to see the shapes at each step:

    python embeddings.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VOCAB_SIZE = 65  # number of unique characters in Tiny Shakespeare
N_EMBD = 32      # embedding dimension (columns in each lookup table)
BLOCK_SIZE = 8   # maximum sequence length (max number of positions)


class EmbeddingLayer(nn.Module):
    """Fuses token identity and position into a single (B, T, C) tensor.

    Args:
        vocab_size: number of distinct tokens (rows in the token table).
        n_embd: embedding dimension (columns in both tables).
        block_size: maximum sequence length (rows in the position table).
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, n_embd: int = N_EMBD,
                 block_size: int = BLOCK_SIZE):
        super().__init__()
        # Lookup table: one learned vector per token identity.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Lookup table: one learned vector per position (0 .. block_size-1).
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx has shape (B, T); returns (B, T, C) with C = n_embd."""
        B, T = idx.shape

        # 1. Token embeddings -> (B, T, C)
        tok_emb = self.token_embedding_table(idx)

        # 2. Position embeddings -> (T, C)
        #    Positions live on the same device as idx so this works on GPU too.
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        # 3. Add them. Broadcasting expands (T, C) across the batch dimension
        #    to line up with (B, T, C).
        x = tok_emb + pos_emb
        return x


# ---------------------------------------------------------------------------
# Demo / self-check
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(1337)

    # --- Token embeddings on their own ---
    token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
    idx = torch.randint(0, VOCAB_SIZE, (4, 8))  # simulate a batch of token IDs
    token_emb = token_embedding_table(idx)
    print(f"Input shape:      {tuple(idx.shape)}")        # (4, 8)
    print(f"Token emb shape:  {tuple(token_emb.shape)}")  # (4, 8, 32)

    # --- Positional embeddings on their own ---
    position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
    pos = torch.arange(0, BLOCK_SIZE, dtype=torch.long)
    pos_emb = position_embedding_table(pos)
    print(f"\nPosition idx shape: {tuple(pos.shape)}")     # (8,)
    print(f"Position emb shape: {tuple(pos_emb.shape)}")   # (8, 32)

    # --- Fused via the reusable module ---
    layer = EmbeddingLayer()
    final_representation = layer(idx)
    print(f"\nFinal representation shape: {tuple(final_representation.shape)}")  # (4, 8, 32)


if __name__ == "__main__":
    main()
