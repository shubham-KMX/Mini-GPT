# Mini-GPT

A decoder-only Transformer (GPT-style language model) built **from scratch** in
PyTorch and trained on the Tiny Shakespeare dataset. Every core component —
tokenizer, embeddings, causal self-attention, LayerNorm, and the training loop
— is implemented by hand rather than pulled from a high-level library, so you
can see exactly how a language model works.

The model learns to generate character-by-character pseudo-Shakespeare:

```
SICINIUS:
And let see your honours.

BRUTUS:
When gentlemen...
```

## How it works

The model predicts **one character at a time**. Given some context, it produces
a probability distribution over the next character, samples one, appends it, and
repeats. This is *autoregressive* generation.

The data flows through the architecture like this:

```
Raw text
  -> Character tokenizer        (text  <-> integer IDs)
  -> Token + positional embeddings
  -> N x Transformer blocks     (causal self-attention + feed-forward, pre-norm + residuals)
  -> Final LayerNorm
  -> Linear head                (project to vocabulary size)
  -> Softmax                    (next-character probabilities)
```

## Project structure

Each file maps to one stage of the build and is runnable on its own (each has a
self-check `main()` that prints its output/shapes).

| File | What it does |
|------|--------------|
| `data.py` | Downloads Tiny Shakespeare, builds a character-level tokenizer, splits into train/val, samples batches |
| `embeddings.py` | Token embeddings + learned positional embeddings, fused by addition |
| `attention.py` | Scaled dot-product causal `Head` and `MultiHeadAttention` |
| `transformer_block.py` | `FeedForward`, from-scratch `LayerNorm`, and a pre-norm residual `Block` |
| `model.py` | `GPTLanguageModel`: assembles everything + cross-entropy loss + `generate()` |
| `train.py` | Training loop (AdamW), periodic train/val loss estimation, sample generation, saves a checkpoint |
| `generate.py` | Loads a trained checkpoint and samples text from your own prompt (no retraining) |
| `Mini_GPT_Colab.ipynb` | Notebook to train on a free Google Colab GPU |

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

### Train on your machine (CPU)

```bash
python train.py                                    # default config (~3000 iters)
python train.py --max-iters 500 --eval-interval 100  # quick test
```

Training saves a checkpoint to `mini_gpt.pt`. On CPU each iteration takes
roughly half a second, so the default run is ~25-30 minutes.

### Train on a GPU (recommended)

Open the notebook on Colab, set the runtime to GPU, and Run all:

```
https://colab.research.google.com/github/shubham-KMX/Mini-GPT/blob/main/Mini_GPT_Colab.ipynb
```

A ~4.8M-parameter model (256-dim, 6 layers, 6 heads) trains in a few minutes on
a free T4 and reaches a validation loss around 1.5.

### Generate text from a trained checkpoint

```bash
python generate.py --prompt "ROMEO:" --tokens 500
python generate.py --prompt "To be or not" --tokens 300 --seed 42
```

- `--prompt` — the seed text the model continues from
- `--tokens` — how many characters to generate
- `--seed` — fix the randomness for reproducible output
- `--checkpoint` — which checkpoint file to load (default `mini_gpt.pt`)

> Only characters that appear in the Tiny Shakespeare vocabulary (letters, space,
> and common punctuation) are understood. Unknown characters in a prompt are
> skipped rather than causing an error.

## Configuration

`train.py` exposes the main hyperparameters as command-line flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--batch-size` | 32 | Sequences processed in parallel |
| `--block-size` | 64 | Context length (max tokens the model looks back on) |
| `--n-embd` | 128 | Embedding dimension |
| `--n-head` | 4 | Number of attention heads |
| `--n-layer` | 4 | Number of Transformer blocks (depth) |
| `--dropout` | 0.1 | Dropout probability |
| `--learning-rate` | 3e-4 | AdamW learning rate |
| `--max-iters` | 3000 | Training iterations |
| `--eval-interval` | 300 | How often to print train/val loss |

## Notes on training

- At initialization the loss should be about `ln(vocab_size)` = `ln(65) ≈ 4.17`
  — a good sanity check that the model is set up correctly.
- Watch **both** train and validation loss. If train loss keeps dropping while
  val loss flattens or rises, the model is overfitting (memorizing rather than
  generalizing). The dataset is small (~1M characters), so more iterations is
  not always better — increasing model size is usually more effective than
  training far past ~5000-10000 iterations.

## Credits

Built as a from-scratch learning project, following the classic character-level
GPT approach popularized by Andrej Karpathy's nanoGPT / makemore. The Tiny
Shakespeare dataset is Karpathy's concatenation of Shakespeare's works.
