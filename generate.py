"""
Generate text from a trained Mini-GPT checkpoint.
=================================================

Training (train.py) saves everything needed to reload the model into a single
checkpoint file (default: mini_gpt.pt) -- the learned weights, the config used
to build the model, and the character vocabulary. This script loads that
checkpoint and samples text WITHOUT retraining.

Examples:
    python generate.py                          # start from a newline
    python generate.py --prompt "ROMEO:"        # continue from a prompt
    python generate.py --prompt "ROMEO:" --tokens 800
    python generate.py --checkpoint mini_gpt.pt --tokens 500
"""

from __future__ import annotations

import argparse

import torch

from model import GPTLanguageModel


def load_model(checkpoint_path: str, device: str):
    """Load a trained model + tokenizer info from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    chars = ckpt["vocab"]

    # Rebuild the char <-> int mappings from the saved vocabulary.
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(ids: list[int]) -> str:
        return "".join(itos[i] for i in ids)

    model = GPTLanguageModel(
        vocab_size=len(chars),
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # turn off dropout etc. -- we are only sampling now

    return model, encode, decode


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a Mini-GPT checkpoint.")
    parser.add_argument("--checkpoint", default="mini_gpt.pt",
                        help="Path to the saved checkpoint (default: mini_gpt.pt).")
    parser.add_argument("--prompt", default="\n",
                        help="Seed text to continue from (default: a newline).")
    parser.add_argument("--tokens", type=int, default=500,
                        help="How many characters to generate (default: 500).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for reproducible output.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, encode, decode = load_model(args.checkpoint, device)

    # Turn the prompt into token ids, skipping any character not in the
    # vocabulary so an unusual prompt does not crash generation.
    known = []
    for c in args.prompt:
        try:
            known.extend(encode(c))
        except KeyError:
            pass
    if not known:
        known = encode("\n")

    context = torch.tensor([known], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.tokens)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
