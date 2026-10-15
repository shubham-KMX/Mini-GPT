# Mini-GPT Project Introduction — Interview Script

Interview mein jab puchhe "tell me about your project" toh yeh bolna. Do versions hain — short (30 sec) aur detailed (2-3 min).

---

## Short Version (30 seconds)

"So I built a mini GPT from scratch in PyTorch — a character-level language model that generates Shakespeare-like text. The model has 4.8 million parameters with 6 Transformer layers. I implemented everything by hand — the tokenizer, token and positional embeddings, causal self-attention with multi-head, LayerNorm, feed-forward networks, and residual connections. Trained it on the Tiny Shakespeare dataset — about 1.1 million characters — on a Google Colab T4 GPU. Started with a loss of 4.38 which is basically random guessing, and brought it down to around 1.7. The model generates coherent text with proper character names, dialogue formatting, and English words. I also built a separate generation script that loads the trained checkpoint and generates text from custom prompts."

---

## Detailed Version (2-3 minutes)

### Part 1: What the project is (15 sec)

"So I built a Mini-GPT — a decoder-only Transformer language model, completely from scratch in PyTorch. No high-level libraries, no pre-built Transformer modules. Everything — the attention mechanism, the normalization, the training loop — I wrote it all by hand. The model learns to generate text that sounds like Shakespeare."

### Part 2: The data pipeline (20 sec)

"It starts with the data. I used the Tiny Shakespeare dataset — about 1.1 million characters of Shakespeare's plays. I built a character-level tokenizer, which means instead of splitting text into words, I split it into individual characters. The vocabulary is just 65 unique characters — a to z, A to Z, space, and some punctuation. I convert the entire text into a sequence of integers, split 90-10 into train and validation sets, and then sample random chunks for training."

### Part 3: Embeddings (20 sec)

"Those integer IDs can't go directly into a neural network — the model would think integer 42 is twice as large as 21, which is meaningless. So I use an embedding layer — basically a lookup table where each character ID maps to a 256-dimensional vector. I also add positional embeddings, because Transformers process everything in parallel and have no sense of order. Without position info, 'the king killed the man' and 'the man killed the king' would look identical to the model. I add both embeddings together to get a representation that knows both WHAT each token is and WHERE it sits."

### Part 4: Self-attention (30 sec)

"Then comes the core — self-attention. Every token generates three vectors: a Query (what am I looking for), a Key (what do I contain), and a Value (what will I share if you attend to me). The attention score is the dot product of Query and Key — if they're aligned, the score is high. I scale these scores by 1 over root of head size to keep them in a stable range, then apply a causal mask — which means each token can only look at itself and previous tokens, never future ones. This is critical because we're predicting the next character, so seeing the future would be cheating. After softmax, these become weights, and I take a weighted sum of the Values. So each token ends up with a new representation that's a mix of information from all the tokens it attended to."

### Part 5: Multi-head attention (15 sec)

"Instead of doing this once, I do it 6 times in parallel — that's multi-head attention. Each head operates in its own 42-dimensional subspace, so different heads can focus on different things — one might learn grammar, another might learn spacing, another might track long-range context. I concatenate all 6 outputs and project them back to 256 dimensions."

### Part 6: Transformer block (25 sec)

"This attention is wrapped inside a Transformer block along with two more things. First, a feed-forward network — this is where each token processes what it gathered during attention. It expands the dimension 4 times to 1024, applies ReLU for non-linearity, then projects back to 256. Second, residual connections — instead of replacing the input with the output, I add them: x plus f of x. This creates a gradient highway so that even in deep networks, gradients can flow back without vanishing. I also apply LayerNorm before each sub-layer — this is called pre-norm, which is the modern GPT style. It keeps training stable. I stack 6 of these blocks on top of each other — each one building on what the previous one learned."

### Part 7: Output and training (25 sec)

"After the 6 blocks, I apply a final LayerNorm and then a linear layer that projects from 256 dimensions back to 65 — one score for each character in the vocabulary. These are called logits. I use cross-entropy loss to measure how wrong the predictions are compared to the actual next characters. The loss starts at around 4.38, which is basically ln of 65 — that's what you'd get with random guessing. I train with AdamW optimizer for 5000 iterations on a T4 GPU. The loss drops to around 1.7, and the model starts generating text that looks like actual Shakespeare — with character names, colons, dialogue formatting, and real English words."

### Part 8: Generation (15 sec)

"For generation, the model takes a prompt, predicts the next character, appends it, and feeds the whole thing back in — this loops for however many characters you want. I crop the context to 256 tokens because the positional embeddings only go up to 256. I also built a separate script that loads the saved checkpoint and generates text from any custom prompt without retraining. The whole project is on GitHub."

---

## Quick architecture flow (if they ask to draw on whiteboard)

```
Raw text (1.1M chars)
    ↓
Character tokenizer (65 unique chars)
    ↓
Integer IDs → PyTorch tensor
    ↓
Token embedding (65 × 256) + Position embedding (256 × 256)
    ↓
Add them → (Batch, 256, 256)
    ↓
┌─────────────────────────────────┐
│  Transformer Block (×6)          │
│                                  │
│  LayerNorm → Multi-Head Attention│
│       ↓                          │
│  Residual add (x + attn(ln(x)))  │
│       ↓                          │
│  LayerNorm → Feed-Forward (4x)   │
│       ↓                          │
│  Residual add (x + ffwd(ln(x)))  │
└─────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
Linear head (256 → 65) = logits
    ↓
Cross-entropy loss (vs actual next char)
    ↓
AdamW backprop → update weights
    ↓
Generation: softmax → sample → append → repeat
```

---

## Key numbers to drop during intro

- 4.8 million parameters
- 6 layers, 6 heads, 256 embedding dim
- 65 character vocabulary
- 1.1 million characters of training data
- Loss: 4.38 (random) → ~1.7 (trained)
- 5000 iterations on T4 GPU
- Context length: 256 tokens
- 97.9% of parameters are in the Transformer blocks
