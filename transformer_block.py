"""
Building the Engine: The Transformer Block
===========================================

Self-attention only *routes* information between tokens; it does not process
that information deeply on its own. A full Transformer block wraps attention
together with three more ingredients:

    FeedForward : a position-wise MLP so each token can "think" about what it
                  gathered during attention (expand x4, ReLU, project back).
    LayerNorm   : standardizes activations to mean 0 / var 1 across the
                  embedding dimension, keeping training numerically stable.
    Residuals   : x + sublayer(x) creates a "gradient highway" so deep stacks
                  train without the signal vanishing.

We use the modern GPT-style "pre-norm" arrangement: LayerNorm is applied
*before* each sub-layer, and the residual bypasses both the norm and the
sub-layer:

    x = x + attn(ln_1(x))
    x = x + ffwd(ln_2(x))

Attention is reused from attention.py (MultiHeadAttention).

Run this file to check that a block preserves the (B, T, C) shape:

    python transformer_block.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import MultiHeadAttention

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_EMBD = 32
N_HEADS = 4
BLOCK_SIZE = 8
DROPOUT = 0.0


class FeedForward(nn.Module):
    """Position-wise MLP: expand by 4x, apply ReLU, then project back down.

    Applied to every token independently and identically -- after attention
    mixes information across positions, this is where each position processes
    it.
    """

    def __init__(self, n_embd: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LayerNorm(nn.Module):
    """LayerNorm implemented from scratch (equivalent to nn.LayerNorm).

    Normalizes across the last (embedding) dimension to mean 0 / var 1, then
    applies a learned scale (gamma) and shift (beta) so the network can undo
    or reshape the normalization when useful.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class Block(nn.Module):
    """A single pre-norm Transformer block: attention + feed-forward, each
    wrapped in a residual connection.

    Args:
        n_embd: embedding dimension.
        n_heads: number of attention heads (must divide n_embd evenly).
        block_size: maximum context length (for the causal mask).
    """

    def __init__(self, n_embd: int = N_EMBD, n_heads: int = N_HEADS,
                 block_size: int = BLOCK_SIZE, dropout: float = DROPOUT):
        super().__init__()
        head_size = n_embd // n_heads
        self.ln_1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(
            num_heads=n_heads, head_size=head_size,
            n_embd=n_embd, block_size=block_size, dropout=dropout,
        )
        self.ln_2 = LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual around attention (norm applied to x before attention).
        x = x + self.attn(self.ln_1(x))
        # Residual around the feed-forward network.
        x = x + self.ffwd(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Demo / self-check
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(1337)

    B, T, C = 4, 8, N_EMBD
    x = torch.randn(B, T, C)

    # --- LayerNorm produces mean ~0, var ~1 across the embedding dim ---
    ln = LayerNorm(C)
    ln_out = ln(x)
    print(f"LayerNorm output mean (~0): {ln_out.mean().item():.6f}")
    print(f"LayerNorm output var  (~1): {ln_out.var(dim=-1, unbiased=False).mean().item():.6f}")

    # --- FeedForward preserves shape ---
    ff = FeedForward(C)
    print(f"\nFeedForward: {tuple(x.shape)} -> {tuple(ff(x).shape)}")

    # --- Full block preserves (B, T, C) ---
    block = Block(n_embd=C, n_heads=N_HEADS, block_size=BLOCK_SIZE)
    out = block(x)
    print(f"Block:       {tuple(x.shape)} -> {tuple(out.shape)}")

    # --- Stacking blocks to build depth ---
    deep = nn.Sequential(*[Block(C, N_HEADS, BLOCK_SIZE) for _ in range(3)])
    deep_out = deep(x)
    print(f"3 stacked:   {tuple(x.shape)} -> {tuple(deep_out.shape)}")


if __name__ == "__main__":
    main()
