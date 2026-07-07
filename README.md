# 🍳 TinyChef — Small Language Model for Recipe Generation

> A Small Language Model trained from scratch on 2.23 million recipes, demonstrating that a 30M-parameter decoder only Transformer can effectively learn recipe-specific language patterns without internet-scale data.

---

## 📋 Project Overview

TinyChef is a domain-constrained Small Language Model (SLM) built entirely from scratch using PyTorch. The core idea is simple: instead of training a massive model on all of the internet, restrict the domain to a single highly structured task — recipe generation — and train a much smaller model that learns language structure within that domain.

The model was trained on 2.23 million recipes from the [RecipeNLG](https://recipenlg.cs.put.poznan.pl/) corpus and achieved a **Test Loss of 2.68**, **Perplexity of 14.54**, and an **average BLEU score of 0.275**, demonstrating its ability to generate coherent and well-structured recipes.

---

## 🏆 Results

| Metric | Value |
|---|---:|
| Test Loss | **2.6767** |
| Perplexity | **14.54** |
| Average BLEU Score | **0.275** |
| Total Parameters | **30,044,544 (~30M)** |

The model was trained for **25,000 iterations** using mixed-precision training, 32-step gradient accumulation, AdamW optimization, and cosine learning-rate decay. Throughout training, the training and validation losses remained closely aligned, indicating stable convergence with minimal overfitting.

---

## 🧠 Model Architecture

TinyChef is a **decoder-only Transformer** designed for autoregressive next-token prediction. The model learns to generate recipes by predicting one token at a time while attending only to previously generated tokens through causal masking.

| Parameter | Value |
|---|---:|
| Transformer Layers | 6 |
| Attention Heads | 6 |
| Embedding Dimension | 384 |
| Context Window | 256 tokens |
| Vocabulary Size | 50,257 (GPT-2 BPE) |
| Dropout | 0.1 |
| Total Trainable Parameters | **~30M** |

Key architectural components:
- Causal masked multi-head self-attention
- Feed-forward layers with GELU activation
- Residual skip connections + Layer Normalization

---

## 📦 Dataset

**RecipeNLG** — A corpus of **2.23 million cooking recipes** introduced in ACL INLG 2020.

| Split | Recipes |
|---|---:|
| Train | **2,205,142** |
| Validation | **21,000** |
| Test | **5,000** |

The dataset consists of semi-structured recipe records containing recipe titles, ingredients, and cooking instructions. Each recipe was converted into a flat autoregressive text format and terminated with a custom `<|endofrecipe|>` token to enable sequential language modeling.

### Getting the Dataset

1. Visit the official RecipeNLG website: **[https://recipenlg.cs.put.poznan.pl/](https://recipenlg.cs.put.poznan.pl/)**
2. Accept the terms (non-commercial research use only)
3. Download `full_dataset.csv`

> ⚠️ The RecipeNLG dataset is licensed for **non-commercial research and educational use only**. This project inherits that restriction.

Each recipe row was converted from structured CSV fields into a flat autoregressive text format:

```
Recipe: No-Bake Nut Cookies
Ingredients: 1 c. brown sugar, 1/2 c. evaporated milk, ...
Instructions:
1. In a heavy saucepan, mix brown sugar, nuts, milk and butter.
2. Stir over medium heat until mixture bubbles.
3. Boil and stir 5 minutes more.
<|endofrecipe|>
```

---

## ⚙️ Training Pipeline

### 1. Tokenization
- GPT-2 Byte Pair Encoding (BPE) tokenizer using `tiktoken`
- Each recipe converted into an autoregressive text sequence ending with `<|endofrecipe|>`
- Token IDs stored as memory-mapped binary (`.bin`) files using `numpy.memmap` for efficient disk-based loading
- Token IDs saved as `uint16` to reduce storage while supporting the GPT-2 vocabulary of 50,257 tokens
- Random mini-batches sampled directly from the binary files during training without loading the complete dataset into memory

### 2. Training Configuration

| Hyperparameter | Value |
|---|---:|
| Max Iterations | **25,000** |
| Mini Batch Size | **32** |
| Effective Batch Size | **1,024 (32 × 32)** |
| Context Window | **256 tokens** |
| Gradient Accumulation Steps | **32** |
| Initial Learning Rate | **1e-4** |
| Minimum Learning Rate | **1e-5** |
| Warmup Steps | **1,000** |
| Mixed Precision | **bfloat16 (if supported), otherwise float16** |
| Optimizer | **AdamW (β₁=0.9, β₂=0.95, weight decay=0.1)** |
| Gradient Clipping | **max_norm = 0.5** |
| Random Seed | **42** |

### 3. Optimisation Tricks
- **Memory-mapped datasets:** Streamed tokenized recipes directly from disk using `numpy.memmap`, enabling training on over 2.2 million recipes without exhausting RAM.
- **Mixed Precision Training:** Used automatic mixed precision (`bfloat16`/`float16`) to reduce GPU memory usage and accelerate computation.
- **Gradient Accumulation (32 steps):** Simulated an effective batch size of 1,024 sequences while fitting within GPU memory constraints.
- **Linear Warmup + Cosine Learning Rate Decay:** Gradually increased the learning rate during the initial training phase before smoothly decaying it for stable convergence.
- **Gradient Clipping:** Limited gradient norms to improve numerical stability and prevent exploding gradients during optimization.


### 4. Infrastructure
- Trained on **Kaggle** using **NVIDIA Tesla T4 ×2 GPUs**
- Best-performing model automatically saved based on validation loss
- Model checkpoints saved every **5,000 iterations**
- Training and validation loss histories periodically saved as `.npy` files for recovery and visualization

---

## 📈 Training Progress

The model was trained for **25,000 iterations** using a 256-token context window. Training and validation losses decreased steadily throughout training, indicating stable convergence with minimal overfitting.

| Iteration | Train Loss | Validation Loss |
|---:|---:|---:|
| 5,000 | 4.1878 | 4.2013 |
| 10,000 | 3.2166 | 3.2274 |
| 15,000 | 2.8787 | 2.8847 |
| 20,000 | 2.7320 | 2.7536 |
| 24,000 | 2.6844 | 2.6917 |
| 24,500 | 2.6882 | **2.6917** |

The close alignment between the training and validation losses throughout training indicates stable optimization and good generalization on unseen recipes.

---

## 🔥 Inference Example

### Prompt
```text
Recipe: Chocolate Chip Cookies
Ingredients:
```

### Generated Output
```text
Recipe: Chocolate Chip Cookies
Ingredients:
2 c. shortening, 1 c. pecans, 1 c. sugar, 1 tsp. soda,
3 tsp. cinnamon, 2 c. baking powder, 2 c. flour,
1/4 c. chopped nuts, 2 eggs, 2 tsp. cinnamon,
1 tsp. baking soda, 2 tsp. salt, 3 c. vanilla extract,
1 c. peanut butter, 1 c. grated margarine,
3 c. brown sugar

Instructions:
1. Mix mixture, eggs, sugar, vanilla, and vanilla.
2. Beat in a 9 × 13-inch dish.
3. Bake at 350°F for 30–40 minutes.
4. Pour cream cheese batter into a greased pan.
5. Sprinkle remaining sugar over the cake.
6. Bake at 350°F for another 30 minutes.
<|endofrecipe|>
```

**Observations**
- ✅ Correct `Recipe → Ingredients → Instructions → <|endofrecipe|>` structure.
- ✅ Fluent procedural language with realistic cooking terminology.
- ✅ Generates plausible ingredient quantities, temperatures, and cooking steps.
- ⚠️ Occasionally hallucinates ingredients or repeats certain items (e.g., cinnamon, vanilla), reflecting the limitations of a compact domain-specific language model.

---

## ✅ Strengths & ⚠️ Limitations

### Strengths
- Consistently generates the expected `Recipe → Ingredients → Instructions → <|endofrecipe|>` format.
- Produces grammatically fluent and coherent cooking instructions.

### Limitations
- Occasional ingredient hallucination or repetition.
- Ingredient lists and cooking instructions are not always perfectly consistent.
- Long-range dependencies can still weaken in longer generations despite the 256-token context window.

---

## 🛠️ How to Run

```python
# Install dependencies
pip install tiktoken torch numpy tqdm pandas

# Inference with trained model
import torch
import tiktoken

enc = tiktoken.get_encoding("gpt2")

model = GPT(config)
model.load_state_dict(torch.load('best_model_params.pt', map_location='cpu'))
model.eval()

prompt = "Recipe: Chocolate Chip Cookies\nIngredients:"
context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0)
output = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=50)
print(enc.decode(output.squeeze().tolist()))
```
---

## 📚 References

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) — Eldan & Li, 2023
- [RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation](https://recipenlg.cs.put.poznan.pl/) — Bień et al., ACL INLG 2020
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
