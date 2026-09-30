"""
Teaching the Machine (Part 1): The GPT Language Model
=====================================================

This assembles every earlier piece into a complete decoder-only Transformer:

    token + positional embeddings   (embeddings.py concepts, inlined here)
    -> a stack of pre-norm Blocks   (transformer_block.py)
    -> final LayerNorm
    -> lm_head : Linear(n_embd -> vocab_size)  producing next-token logits

The forward pass optionally computes cross-entropy loss against target tokens,
and `generate` performs autoregressive sampling.

A useful sanity check: at initialization the loss should be about
ln(vocab_size). For vocab_size = 65 that is ~4.17 -- roughly the entropy of a
uniform guess over the vocabulary.

Run this file to construct the model, print its size, and confirm the initial
loss and a generation pass both work:

    python model.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer_block import Block


class GPTLanguageModel(nn.Module):
    """A small GPT-style character-level language model."""

    def __init__(self, vocab_size: int, n_embd: int, block_size: int,
                 n_layer: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size

        # 1. Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 2. Stack of Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        # 3. Final layer norm + language-modeling head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx and targets are (B, T) integer tensors.

        Returns (logits, loss). loss is None when targets is None.
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, n_embd)
        x = tok_emb + pos_emb          # (B, T, n_embd)  broadcast over batch
        x = self.blocks(x)             # (B, T, n_embd)
        x = self.ln_f(x)               # (B, T, n_embd)
        logits = self.lm_head(x)       # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # F.cross_entropy wants (N, C) logits and (N,) targets, so flatten
            # the batch and time dimensions together.
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively extend idx (B, T) by max_new_tokens tokens."""
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens (positional embeddings
            # and the causal mask are only defined up to block_size).
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)          # (B, T, vocab_size)
            logits = logits[:, -1, :]           # focus on the last step (B, C)
            probs = F.softmax(logits, dim=-1)   # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx


# ---------------------------------------------------------------------------
# Demo / self-check
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(1337)

    vocab_size = 65
    model = GPTLanguageModel(
        vocab_size=vocab_size, n_embd=32, block_size=8, n_layer=3, n_head=4,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Initial-loss sanity check: should be near ln(vocab_size).
    xb = torch.randint(0, vocab_size, (4, 8))
    yb = torch.randint(0, vocab_size, (4, 8))
    logits, loss = model(xb, yb)
    import math
    print(f"Logits shape: {tuple(logits.shape)}")           # (4, 8, 65)
    print(f"Initial loss: {loss.item():.4f}  (expected ~{math.log(vocab_size):.4f})")

    # Generation smoke test: start from a single token, produce 20 more.
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=20)
    print(f"Generated token ids: {generated[0].tolist()}")


if __name__ == "__main__":
    main()
