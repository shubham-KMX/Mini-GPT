"""
Prepare a Hinglish training corpus for the mini model.
======================================================

By default this pulls the `romanized_casual` field from ai4bharat/IndicCMix
(the Hindi split, hi.parquet) -- WhatsApp-style Roman-script Hinglish. That is
a clean, MIT-licensed source and a good size match for the small model. The
script:

    1. Streams the dataset (does NOT download everything).
    2. Cleans each line -- removes URLs, @mentions, and # symbols, collapses
       whitespace, drops near-empty lines.
    3. Writes cleaned lines to input.txt until it reaches the target size.

IMPORTANT -- IndicCMix is a *gated* dataset:
    You must accept its terms once on the dataset page and log in from the
    machine running this script. On Colab:

        from huggingface_hub import login
        login()   # paste a token from https://huggingface.co/settings/tokens

    Then accept the terms at https://huggingface.co/datasets/ai4bharat/IndicCMix

    Any other Hugging Face text dataset works too -- point --dataset /
    --data-files / --text-field at it.

Usage:
    pip install datasets
    python prepare_hinglish.py                                    # ~5 MB -> input.txt
    python prepare_hinglish.py --dataset <id> --text-field <col>  # a different dataset
    python prepare_hinglish.py --target-mb 3 --out mydata.txt     # smaller / custom file

By default this OVERWRITES input.txt so the rest of the pipeline uses the
Hinglish corpus automatically. Delete input.txt to get Shakespeare back (it
re-downloads on the next run).

Then train on it (no --data needed, since input.txt is the default corpus):
    python train.py --block-size 256 --n-embd 256 \
        --n-head 6 --n-layer 6 --max-iters 5000

Dataset: ai4bharat/IndicCMix (MIT license).
"""

from __future__ import annotations

import argparse
import re

# URLs, @mentions, and the '#' symbol (we keep the hashtag word itself).
_URL = re.compile(r"http\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASH = re.compile(r"#")
_WS = re.compile(r"\s+")


def clean_line(text: str) -> str:
    """Strip tweet noise and normalize whitespace."""
    text = _URL.sub("", text)
    text = _MENTION.sub("", text)
    text = _HASH.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Hinglish corpus from L3Cube-HingCorpus.")
    # Default output is input.txt so the rest of the pipeline (data.py,
    # train.py) picks up the Hinglish corpus with no extra flags. This
    # overwrites any existing Shakespeare input.txt; if you later want
    # Shakespeare back, just delete input.txt and it re-downloads automatically.
    parser.add_argument("--out", default="input.txt", help="Output text file.")
    parser.add_argument("--target-mb", type=float, default=5.0,
                        help="Approximate output size in megabytes (default: 5).")
    parser.add_argument("--min-chars", type=int, default=15,
                        help="Skip lines shorter than this many characters.")
    parser.add_argument("--dataset", default="ai4bharat/IndicCMix",
                        help="Hugging Face dataset id to stream from.")
    parser.add_argument("--data-files", default="hi.parquet",
                        help="Specific file(s) within the dataset repo "
                             "(IndicCMix ships one parquet per language). "
                             "Set to empty string to load the default split.")
    parser.add_argument("--text-field", default="romanized_casual",
                        help="Which column holds the Hinglish text "
                             "(IndicCMix uses 'romanized_casual').")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' library is required. Install it with:\n"
            "    pip install datasets"
        )

    target_bytes = int(args.target_mb * 1024 * 1024)
    print(f"Streaming {args.dataset} ({args.data_files or 'default'}) "
          f"-> {args.out} (target ~{args.target_mb} MB)")

    # Stream so we never download the entire dataset.
    load_kwargs = dict(split="train", streaming=True)
    if args.data_files:
        load_kwargs["data_files"] = args.data_files
    ds = load_dataset(args.dataset, **load_kwargs)

    written = 0
    kept = 0
    seen = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            seen += 1
            # Prefer the configured text field; fall back to 'text', then to
            # the first string value in the row.
            raw = row.get(args.text_field) or row.get("text")
            if raw is None:
                raw = next((v for v in row.values() if isinstance(v, str)), "")
            line = clean_line(raw)
            if len(line) < args.min_chars:
                continue
            line += "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            kept += 1
            if kept % 20000 == 0:
                print(f"  kept {kept:,} lines | {written/1024/1024:.2f} MB")
            if written >= target_bytes:
                break

    print(f"\nDone. Scanned {seen:,} rows, kept {kept:,} lines, "
          f"wrote {written/1024/1024:.2f} MB to {args.out}")


if __name__ == "__main__":
    main()
