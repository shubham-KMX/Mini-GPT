"""
Prepare a Hinglish training corpus from L3Cube-HingCorpus.
==========================================================

L3Cube-HingCorpus is a large (1.04B token) Hindi-English code-mixed corpus in
Roman script, scraped from Twitter. That is far larger than our small model
needs, and the raw tweets are noisy, so this script:

    1. Streams the dataset (does NOT download all ~1B tokens).
    2. Cleans each line -- removes URLs, @mentions, and #hashtag symbols,
       collapses whitespace, drops near-empty lines.
    3. Writes cleaned lines to hinglish.txt until it reaches the target size
       (default ~5 MB), which is a good match for the mini model.

IMPORTANT -- dataset id:
    L3Cube publishes HingCorpus mainly via their GitHub / Google Drive; the
    exact Hugging Face *dataset* id can vary. Pass the correct one with
    --dataset if the default fails. Any Hugging Face text dataset with a
    string column also works (e.g. another Hinglish corpus). Best run on
    Google Colab, where the `datasets` library and bandwidth are available.

Usage:
    pip install datasets
    python prepare_hinglish.py                                   # ~5 MB -> input.txt
    python prepare_hinglish.py --dataset <hf_dataset_id>         # if default fails
    python prepare_hinglish.py --target-mb 3 --out mydata.txt    # smaller / custom file

By default this OVERWRITES input.txt so the rest of the pipeline uses the
Hinglish corpus automatically. Delete input.txt to get Shakespeare back (it
re-downloads on the next run).

Then train on it (no --data needed, since input.txt is the default corpus):
    python train.py --block-size 256 --n-embd 256 \
        --n-head 6 --n-layer 6 --max-iters 5000

Reference: L3Cube-HingCorpus, https://github.com/l3cube-pune/code-mixed-nlp
(dataset details rephrased for licensing compliance).
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
    parser.add_argument("--dataset", default="l3cube-pune/HingCorpus",
                        help="Hugging Face dataset id to stream from.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' library is required. Install it with:\n"
            "    pip install datasets"
        )

    target_bytes = int(args.target_mb * 1024 * 1024)
    print(f"Streaming {args.dataset} -> {args.out} (target ~{args.target_mb} MB)")

    # Stream so we never download the full ~1B-token corpus.
    ds = load_dataset(args.dataset, split="train", streaming=True)

    written = 0
    kept = 0
    seen = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            seen += 1
            # The text column is usually 'text'; fall back to the first string field.
            raw = row.get("text")
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
