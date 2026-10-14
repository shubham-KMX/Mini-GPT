# Mini-GPT Interview Prep — Layman Answers

Har sawaal ka jawab simple, casual language mein. Interview mein bhi aise hi bolna — complicated words mat daalna jab simple se kaam chal jaye.

---

## 1. Tokenization (data.py)

### Q: Why character-level and not BPE?

Shakespeare mein sirf 65 unique characters hain — a-z, A-Z, space, punctuation. Itna chhota vocab hai ki character-level best choice hai. Simple hai, fast hai, aur learning ke liye perfect.

BPE (Byte Pair Encoding) tab use hota hai jab vocab bahut bada ho — jaise GPT-2 mein 50,000 tokens. BPE "common", "ing", "pre" jaise subword pieces banata hai. Zabardast hai par complex. Hum yahan basics samajhna chahte hain, isliye character-level.

### Q: What happens if a character in your prompt is not in the vocabulary?

generate.py mein try-except hai. Agar character vocab mein nahi hai (jaise emoji ya Hindi letter), toh woh character skip ho jata hai. Model crash nahi hota. Baaki prompt chal jata hai.

### Q: What's the relationship between vocab size and sequence length?

Inverse relationship. Vocab chhota (65 chars) → sequence bahut lamba (1 million characters). Vocab bada (50k BPE tokens) → sequence chhota (~100k tokens). Same data, bas packaging alag.

Chhota vocab ka fayda: model har character ka representation seekh jata hai. Nuksan: ek word banane ke liye bahut saare characters predict karne padte hain.

---

## 2. Embeddings (embeddings.py + model.py)

### Q: Why do we need token embeddings?

Neural network numbers samajhta hai. Agar hum seedha integer 42 daalein, toh network sochega "42 = 2 x 21" — ki 21 se double hai. But 42 sirf ek arbitrary ID hai, "e" character ka. Iska 21 se koi mathematical relation nahi hai.

Embedding ek lookup table hai. Har integer ID ka apna 256-dimensional vector hai. Training ke dauran ye vectors adjust hote hain — similar characters ki vectors similar ho jaati hain. Network khud seekh jata hai ki "a" aur "e" zyada similar hain "a" aur "Z" se.

### Q: Why add token and position embeddings instead of concatenating?

Agar concatenate karte toh dimension badh jati — 256 + 256 = 512. Model bada ho jata, zyada parameters, zyada compute.

Add karne se dimension same rehti hai (256). Token identity aur position dono ka information ek hi vector mein pack ho jata hai. Network baad mein dono ko alag kar seekh jata hai. Same result, kam memory.

### Q: Why learned positional embeddings vs sinusoidal?

Original Transformer paper (2017) mein sine/cosine functions the — fixed mathematical formulas. Bahut complex.

Humne learned embeddings use kiye — simple nn.Embedding table. Har position (0, 1, 2...255) ka apna learned vector hai. Model khud seekh jata hai ki position 5 kya matlab hai.

Chhote models ke liye learned behtar kaam karta hai. Bade models mein bhi ye standard ho gaya hai (GPT-2, GPT-3 dono learned use karte hain).

### Q: What's the shape change: (B,T) -> (B,T,C)?

Input: (Batch, Time) = (64, 256) — 64 sequences, har sequence 256 tokens.
Output: (Batch, Time, Channels) = (64, 256, 256) — har token 256-dim vector mein badal gaya.

Har integer ek 256-dim vector ban gaya. Itna hi — ek extra dimension add hui.

---

## 3. Self-Attention (attention.py) — SABSE IMPORTANT

### Q: Walk me through the attention computation step by step

Step 1: Har token 3 vectors banata hai — Query (kya dhoondh raha hoon), Key (kya hai mere paas), Value (main kya de dunga agar attention mile). Ye 3 vectors Linear layers se bante hain.

Step 2: Attention score = Query dot Key. Agar token 5 ka Query token 2 ke Key se match karta hai, toh score high. Har token har dusre token ke saath score banata hai. Matrix ban jata hai (T x T).

Step 3: Scale by 1/sqrt(head_size). Bina iske scores bahut bade ho jate hain aur softmax ek hi token pe focus kar lega.

Step 4: Causal mask. Future tokens ko -inf set kar do. Token 5 token 6,7,8 ko nahi dekh sakta. Sirf 0-4 dekh sakta hai.

Step 5: Softmax. -inf wale 0 ho jate hain. Baaki scores probabilities ban jate hain (row sums to 1).

Step 6: Weights x Values. Har token apni weights ke hisaab se dusre tokens ki Values mix karta hai. Output: har token ka naya contextualized vector.

### Q: Why -inf and not 0 for masking?

Agar 0 daalte, toh softmax usko ek valid score samjhta. exp(0) = 1, yaani future token ko bhi weight mil jata. Data leakage.

-inf daalne se exp(-inf) = 0. Future token ka weight exactly 0. Clean cut, koi cheating nahi.

### Q: Why scale by 1/sqrt(head_size)?

Dot product ka variance head_size ke saath grow karta hai. Agar head_size 42 hai, toh 42 values ka sum aa raha hai. Bade sums = bade scores. Bade scores = softmax ek hi token pe 0.99 de baaki pe 0.01. Gradient vanish.

1/sqrt(42) se divide karne ke scores normal range mein rehte hain. Softmax smooth distribution deta hai. Sab tokens ko fair chance. Training stable.

### Q: What's the shape of the attention weight matrix?

(B, T, T) — Batch x Time x Time. Har sequence ke liye ek T x T matrix. Row i = token i ka attention distribution over all tokens. Column j = token j ko kitne tokens attend kar rahe hain.

### Q: Why not let the model see future tokens?

Ye language model hai — next token predict karna hai. Agar future dekh sakta hai, toh cheating hai. "What comes after 'ROMEO:'?" ka jawal already input mein hai. Model kuch seekhe ga nahi, bas copy karega.

Generation ke time bhi same problem — hum future nahi jante, toh model ko bhi nahi dekhna chahiye during training.

---

## 4. Multi-Head Attention (attention.py)

### Q: Why multiple heads instead of one big head?

Ek head ek tarah ka relationship dekhta hai. Maybe head 1 grammar seekhta hai (adjective-noun), head 2 spacing (punctuation), head 3 long-range context (start of sentence).

6 heads = 6 alag perspectives. Har head 42-dim space mein attention compute karta hai. Saab milke 256-dim ban jata hai. Zyada diverse information capture hota hai.

### Q: How are heads combined?

Simple. 6 heads ke outputs concatenate hote hain along channel dimension: 6 x 42 = 252. Phir ek Linear layer (252 -> 256) se wapas original dimension pe project hota hai. Combine + mix.

### Q: How many heads does your model have and why?

6 heads, 256/6 = 42 dims per head. 6 isliye kyunki 256 cleanly divide hota hai (42 x 6 = 252, thoda remainder but proj layer handle kar leti hai). Common choice hai — GPT-2 small mein bhi similar ratio hai.

---

## 5. Transformer Block (transformer_block.py)

### Q: What does the feed-forward network do that attention doesn't?

Attention sirf information ROUTE karta hai — "token 5 ko token 2 ki info chahiye". But us info ko PROCESS nahi karta.

Feed-forward wo processing karta hai. Attention ke baad har token apne paas gathered info le ke aata hai. FFN us info ko "sochta" hai — expand 4x (1024 dims), ReLU (non-linearity), wapas 256 pe. Per token independently.

Analogy: attention = group meeting mein notes share karna. FFN = meeting ke baad desk pe baith ke notes process karna.

### Q: Why is FFN applied per-position, not across the sequence?

Cross-position mixing attention ka kaam hai. FFN ka kaam within-position processing hai. Agar FFN bhi cross-sequence karta, toh dono same kaam karte — redundant. Architecture ka design yehi hai: attention mixes, FFN computes. Dono alag roles.

### Q: What problem do residual connections solve?

Deep networks mein backpropagation se gradient pehle ke layers tak pahunch hi nahi paata. Har layer pe matrix multiply hota hai, gradient shrink hota jaata hai. Vanishing gradient.

Residual: output = x + f(x). Derivative of x is 1 — gradient seedha flow karta hai, bina shrink hue. Chahe 100 layers ho, gradient pehle layer tak pahunch jayega. "Gradient highway."

### Q: What's the difference between pre-norm and post-norm?

Post-norm (original 2017 paper): x = LayerNorm(x + sublayer(x)). Norm residual ke baad.

Pre-norm (modern GPT style, hamara model): x = x + sublayer(LayerNorm(x)). Norm sublayer se pehle.

Pre-norm behtar kyunki: residual path pe raw x flow karta hai bina norm ke. Training stable, bigger models mein bhi. Post-norm mein deep models unstable ho jate hain.

### Q: Why expand by 4x in the FFN?

256 -> 1024 -> 256. Expand karne se network ke paas zyada "thinking space" hai. 1024 dims mein complex patterns represent ho sakte hain. Phir wapas 256 pe compress ho jata hai — sirf important info rakh ke.

4x standard hai — original paper mein bhi yahi tha. Empirically best balance between capacity aur compute.

---

## 6. Model Architecture (model.py)

### Q: Draw the architecture from input to output

```
Input: (B, T) integer token IDs
  |
  Token Embedding + Position Embedding  ->  (B, T, 256)
  |
  Block 1: [LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual]
  Block 2: same
  Block 3: same
  Block 4: same
  Block 5: same
  Block 6: same
  |
  Final LayerNorm  ->  (B, T, 256)
  |
  LM Head: Linear(256 -> 65)  ->  (B, T, 65)
  |
  Output: logits (raw scores for each of 65 characters at each position)
```

### Q: Why flatten before cross-entropy?

F.cross_entropy 2D input expect karta: (N, classes) aur targets (N,). Humara logits 3D hai: (B, T, vocab). Toh B aur T ko merge kiya: (B*T, vocab). Har position ek independent prediction ban gaya. Same targets ke saath: (B*T,).

### Q: What should the initial loss be and why?

ln(65) = 4.17. Random guessing ka loss yahi hota hai. 65 characters, har ek ka probability 1/65, negative log likelihood = -ln(1/65) = ln(65) = 4.17.

Humara model start pe 4.38 tha — thoda zyada par approx 4.17 ke aas-paas. Agar 10 ya 0 aata toh bug hota. 4.17 = code sahi hai.

### Q: How many total parameters and where are they concentrated?

4.8 million trainable. 97.9% Transformer blocks mein. Har block ~785K params — attention (582K) aur FFN (526K) mein split. Embeddings aur lm_head sirf 2.1%.

Iska matlab: model ka "dimaag" blocks mein hai. Embeddings sirf input format karne ka kaam karti hain.

---

## 7. Training (train.py)

### Q: Why zero gradients every step?

PyTorch gradients ACCUMULATE karta hai. Agar zero nahi karenge, toh is batch ke gradients pehle batch ke gradients ke saath add ho jayenge. Direction galat ho jayega.

zero_grad(set_to_none=True) se purane gradients delete ho jate hain (None set). Thoda fast bhi — memory free ho jati hai.

### Q: How do you detect overfitting?

Train loss dekho, val loss dekho. Agar dono saath gir rahe hain — good, model seekh raha hai. Agar train loss gir rahi par val loss rukk gayi ya badh rahi — overfitting. Model data yaad kar raha, generalize nahi kar raha.

Hamare model mein dono saath-saath gir rahe the (4.38 -> 1.7) — no overfitting.

### Q: Why evaluate on multiple batches and average?

Ek batch ka loss noisy hai — randomly koi easy batch aa jaye toh loss kam, hard batch aa jaye toh zyada. 200 batches ka average stable estimate deta hai. Isliye estimate_loss 200 batches run karta hai phir average nikalta hai.

### Q: What's the role of dropout?

Training ke dauran random neurons ko off kar deta hai (10-20%). Isse model kisi specific neuron pe depend nahi karta — robust ban jata hai. Overfitting kam.

Evaluation pe dropout off ho jata hai (model.eval()). Saare neurons kaam karte hain. Best performance.

---

## 8. Text Generation (model.py + generate.py)

### Q: How does your model generate text?

Autoregressive loop:
1. Input do (jaise "ROMEO:")
2. Model 65 characters ke liye scores nikalta hai
3. Sirf last position ke scores lo (softmax -> probabilities)
4. Us distribution se sample karo (multinomial) — ek character mila
5. Us character ko input mein append karo
6. Wapas step 2 pe jao
7. 500 baar repeat karo

### Q: Why crop the context to block_size?

Positional embeddings sirf 256 positions ke liye defined hain. Agar context 256 se bada ho gaya, toh position 257, 258... ke liye embedding nahi hai. Error.

Isliye hum sirf last 256 tokens rakhte hain: idx[:, -block_size:]. Purana context drop ho jata hai par model ko latest 256 ka context milta hai.

### Q: Why sample from the distribution instead of taking argmax?

Argmax = hamesha highest probability wala character. Boring. Same input = hamesha same output. No variety.

Sampling = distribution se randomly pick. High probability characters zyada baar aate hain, par kabhi-kabhi low probability wala bhi aa jata hai. Har baar different, creative text.

### Q: What's in your checkpoint file?

3 cheezein:
1. model_state — saare learned weights (4.8M parameters)
2. config — model ka architecture (n_embd, n_layer, n_head, etc.)
3. vocab — 65 characters ki list

Isse generate.py model ko bina training ke wapas bana sakta hai aur text generate kar sakta hai.

---

## Bonus Questions

### Q: What would you change to improve output quality?

1. BPE tokenizer instead of character-level — words/subwords seekhega, better quality
2. Bada model — zyada layers (12+), zyada dims (512+)
3. Zyada data — Shakespeare 1MB hai, 100MB+ text do toh better
4. Ziada training iterations — 5000 se 50000 tak (with bigger data)
5. Learning rate scheduling — start high, gradually decrease

### Q: How does your model compare to GPT-2/3?

| | Mine | GPT-2 Small | GPT-3 |
|---|---|---|---|
| Parameters | 4.8M | 124M | 175B |
| Tokenizer | Character (65) | BPE (50K) | BPE (50K) |
| Layers | 6 | 12 | 96 |
| Context | 256 | 1024 | 2048 |
| Training data | 1MB | 40GB | 570GB |

Architecture same hai — bas scale 1000x chhota. Proof: architecture sahi hai, bas data aur compute aur do toh GPT-2 jaisa output aayega.

### Q: What was the hardest part to implement?

Causal self-attention. Teen cheezein simultaneously handle karni thi:
1. Q·K^T matrix multiply with correct dimensions
2. Causal masking (-inf at right places)
3. Scaling factor (1/sqrt(head_size))

Ek shape galat hui toh pura forward pass error. Debug karne mein time laga.

### Q: What did you learn from this project?

1. Transformer ke andar kuch magic nahi hai — sirf matrix multiplies, softmax, aur linear layers
2. Attention ka intuition: har token decides kis ko dekhna hai
3. Residual connections kitne important hain — bina inke deep networks train nahi hote
4. Small model (4.8M) bhi sahi architecture sahi output de sakta hai
5. Loss 4.17 (random) se 1.7 tak dekh ke samajh aaya ki model sach mein seekh raha hai
