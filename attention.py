"""
Looking Back: Causal Self-Attention
====================================

Embeddings tell the model *what* each token is and *where* it sits, but tokens
still need to share context -- the meaning of "bank" depends on whether "river"
or "robbery" came before it. Self-attention is how tokens communicate.

Each token produces three vectors:

    Query (Q): what I am looking for
    Key   (K): what I contain
    Value (V): what I hand over if you attend to me

Attention scores are Q . K^T (how well each query matches each key). We then:

    1. Scale by 1/sqrt(head_size) so softmax stays well-conditioned.
    2. Apply a CAUSAL mask so a token can only attend to itself and the past
       (the future is set to -inf before softmax).
    3. Softmax the scores into weights, then take a weighted sum of the Values.

This file implements a single attention `Head` and `MultiHeadAttention`
(several heads in parallel, concatenated and projected back to n_embd).

Run it to see the shapes and confirm the causal mask works:

    python attention.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_EMBD = 32       # embedding dimension (channels) coming in
BLOCK_SIZE = 8    # maximum sequence length (for the causal mask buffer)
DROPOUT = 0.0     # dropout probability on the attention weights / output


class Head(nn.Module):
    """One head of causal self-attention.

    Projects the input (B, T, n_embd) into Q, K, V of width `head_size`,
    computes scaled dot-product attention with a causal mask, and returns a
    contextualized output of shape (B, T, head_size).
    """

    def __init__(self, head_size: int, n_embd: int = N_EMBD,
                 block_size: int = BLOCK_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower-triangular matrix used to mask out future positions. Registered
        # as a buffer so it moves with the module (.to(device)) but is not a
        # learnable parameter.
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Attention scores ("affinities"): (B, T, T)
        # Scale by 1/sqrt(head_size) to keep the variance ~1 so softmax does
        # not saturate into near-one-hot vectors early in training.
        # (Note: the course text scales by C ** -0.5. The mathematically
        # correct factor is head_size ** -0.5, since the dot product is taken
        # over head_size dimensions -- that is what we use here.)
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size ** -0.5

        # Causal mask: forbid attending to future positions by setting the
        # upper triangle to -inf (softmax turns -inf into 0 weight).
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of the values.
        v = self.value(x)   # (B, T, head_size)
        out = wei @ v       # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Several attention heads running in parallel.

    Each head attends in its own `head_size`-dim subspace; the outputs are
    concatenated back to `num_heads * head_size` (== n_embd) and passed through
    a final linear projection.
    """

    def __init__(self, num_heads: int, head_size: int, n_embd: int = N_EMBD,
                 block_size: int = BLOCK_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate along the channel dimension: (B, T, num_heads*head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# ---------------------------------------------------------------------------
# Demo / self-check
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(1337)

    B, T, C = 4, 8, 32   # batch, time, channels
    head_size = 16
    x = torch.randn(B, T, C)

    # --- Single head ---
    head = Head(head_size)
    out = head(x)
    print(f"Input shape:        {tuple(x.shape)}")    # (4, 8, 32)
    print(f"Single-head output: {tuple(out.shape)}")  # (4, 8, 16)

    # --- Verify the attention weights are truly causal ---
    # Re-derive the weights the same way forward() does, for inspection.
    k = head.key(x)
    q = head.query(x)
    wei = q @ k.transpose(-2, -1) * head_size ** -0.5
    wei = wei.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    upper = wei[0].triu(diagonal=1)  # everything strictly above the diagonal
    print(f"\nAttention weights row 0 (first token attends only to itself):")
    print(wei[0, 0])
    print(f"Max weight in the upper triangle (should be 0): {upper.max().item():.6f}")
    print(f"Each row sums to 1: {torch.allclose(wei[0].sum(-1), torch.ones(T))}")

    # --- Multi-head (4 heads x 8 dims = 32 = n_embd) ---
    mha = MultiHeadAttention(num_heads=4, head_size=8)
    mha_out = mha(x)
    print(f"\nMulti-head output:  {tuple(mha_out.shape)}")  # (4, 8, 32)

    # --- Reproduce the course's verification setup ---
    # n_embd=32, num_heads=4, head_size=8, block_size=16, batch=2
    n_embd, num_heads = 32, 4
    course_head_size = n_embd // num_heads  # 8
    course_mha = MultiHeadAttention(
        num_heads=num_heads, head_size=course_head_size,
        n_embd=n_embd, block_size=16,
    )
    dummy_input = torch.randn(2, 16, n_embd)
    course_out = course_mha(dummy_input)
    print(f"\n[Course example] Input shape:  {tuple(dummy_input.shape)}")   # (2, 16, 32)
    print(f"[Course example] Output shape: {tuple(course_out.shape)}")      # (2, 16, 32)


if __name__ == "__main__":
    main()
