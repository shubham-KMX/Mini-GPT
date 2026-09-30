"""
Teaching the Machine (Part 2): Training and Generation
======================================================

The training loop that turns the randomly-initialized GPTLanguageModel into a
Shakespeare imitator:

    1. Load and tokenize Tiny Shakespeare (data.py).
    2. Instantiate the model (model.py).
    3. Repeatedly: sample a batch -> forward (loss) -> zero grads -> backward
       -> optimizer step (AdamW).
    4. Every `eval_interval` steps, estimate train/val loss over several
       batches with gradients disabled (a reliable, less noisy signal and a
       check for overfitting).
    5. After training, generate a sample of text and decode it back to
       characters.

Usage:
    python train.py                 # train with the default config
    python train.py --max-iters 200 # quick smoke test

On CPU the defaults are intentionally small so this finishes in minutes. Bump
n_embd / n_layer / n_head / block_size for higher quality on a GPU.
"""

from __future__ import annotations

import argparse

import torch

from data import CharTokenizer, build_dataset, get_batch, load_text
from model import GPTLanguageModel


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    batch_size=32,
    block_size=64,      # context length
    n_embd=128,
    n_head=4,
    n_layer=4,
    dropout=0.1,
    learning_rate=3e-4,
    max_iters=3000,
    eval_interval=300,
    eval_iters=200,
    seed=1337,
)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    """Average the loss over `eval_iters` batches for train and val splits."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, batch_size, block_size)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data ---
    text = load_text()
    tokenizer = CharTokenizer(text)
    _, train_data, val_data = build_dataset(text, tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size} | "
          f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    # --- Model ---
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        dropout=cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    # --- Training loop ---
    for it in range(cfg["max_iters"]):
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"] - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                cfg["batch_size"], cfg["block_size"], cfg["eval_iters"], device,
            )
            print(f"Step {it:5d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data,
                           cfg["batch_size"], cfg["block_size"])
        xb, yb = xb.to(device), yb.to(device)

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)  # clear accumulated grads
        loss.backward()
        optimizer.step()

    # --- Generate a sample ---
    print("\n--- Generated sample ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=cfg.get("gen_tokens", 500))
    print(tokenizer.decode(generated[0].tolist()))

    # --- Save checkpoint ---
    ckpt_path = "mini_gpt.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "vocab": tokenizer.chars}, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")


def parse_args() -> dict:
    p = argparse.ArgumentParser(description="Train the Mini-GPT on Tiny Shakespeare.")
    for key, val in DEFAULTS.items():
        arg = "--" + key.replace("_", "-")
        p.add_argument(arg, type=type(val), default=val)
    p.add_argument("--gen-tokens", type=int, default=500,
                   help="How many characters to generate after training.")
    return vars(p.parse_args())


if __name__ == "__main__":
    train(parse_args())
