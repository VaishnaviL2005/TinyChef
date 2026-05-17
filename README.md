# 🍳 TinyChef — Small Language Model for Recipe Generation

> A GPT-style transformer trained from scratch on 2.2 million recipes, proving that a 30M parameter model can learn coherent recipe structure without internet-scale data.

---

## 📋 Project Overview

TinyChef is a domain-constrained Small Language Model (SLM) built entirely from scratch using PyTorch. Inspired by the [TinyStories](https://arxiv.org/abs/2305.07759) research paper, the core idea is simple: instead of training a massive model on all of the internet, restrict the domain to a single highly structured task — recipe generation — and train a much smaller model that learns language structure within that domain.

The model was trained on 2.23 million recipes from the [RecipeNLG](https://recipenlg.cs.put.poznan.pl/) corpus and achieved a final validation loss of **2.0148** — outperforming the course baseline of 2.3919 achieved on an A100 GPU at 20,000 iterations.

---

## 🏆 Results

| Metric | TinyChef (T4 x2, 30K iters) |
|---|---|
| Final Train Loss |**2.0335** |
| Best Validation Loss |**2.0148** |
| Train/Val Gap | <0.019 |
| Parameters | **30M** |

The train/validation gap never exceeded **0.02** across all 30,000 iterations — confirming the model generalised well and did not overfit.

---

## 🧠 Model Architecture

A decoder-only GPT-style transformer with the following configuration:

| Parameter | Value |
|---|---|
| Transformer Layers | 6 |
| Attention Heads | 6 |
| Embedding Dimension | 384 |
| Context Window | 128 tokens |
| Vocabulary Size | 50,257 (GPT-2 BPE) |
| Dropout | 0.1 |
| **Total Parameters** | **~30M** |

Key architectural components:
- Causal masked multi-head self-attention
- Feed-forward layers with GELU activation
- Residual skip connections + Layer Normalization
- Weight-tied token embeddings and output head

---

## 📦 Dataset

**RecipeNLG** — 2.23 million cooking recipes (ACL INLG 2020)

| Split | Recipes |
|---|---|
| Train | 2,210,142 |
| Validation | 21,000 |

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
- GPT-2 BPE tokenizer via `tiktoken`
- Token IDs stored as memory-mapped `.bin` files using `numpy.memmap` to avoid RAM overload with 2.2M recipes

### 2. Training Configuration

| Hyperparameter | Value |
|---|---|
| Max Iterations | 30,000 |
| Batch Size | 32 |
| Block Size (Context) | 128 |
| Gradient Accumulation Steps | 32 |
| Learning Rate | 1e-4 |
| Min LR (cosine decay) | 5e-4 |
| Warmup Steps | 1,000 |
| Precision | float16 (mixed) |
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| Gradient Clipping | max_norm=0.5 |

### 3. Optimisation Tricks
- **Mixed Precision (float16)**: Faster matrix ops on T4 GPU
- **Gradient Accumulation**: Effective batch size of 32×32=1024 without OOM
- **Cosine LR Decay** with linear warmup
- **Gradient Clipping** for stable late-stage training

### 4. Infrastructure
- Trained on Kaggle (T4 x2 GPU) — ~6 hours total
- Periodic checkpoint saving every 2,000 iterations
- Loss lists saved to `.npy` files for plot recovery across sessions

---

## 📈 Loss Curve

| Iteration | Train Loss | Val Loss |
|---|---|---|
| 2,000 | 4.91 | 4.89 |
| 6,000 | 3.37 | 3.37 |
| 10,000 | 2.79 | 2.77 |
| 14,000 | 2.49 | 2.47 |
| 18,000 | 2.29 | 2.28 |
| 22,000 | 2.17 | 2.16 |
| 26,000 | 2.09 | 2.08 |
| 29,500 | 2.03 | **2.01** |

Both curves converge smoothly with no divergence — a sign of healthy training throughout.

---

## 🔥 Inference Examples

### Prompt 1: Chocolate Chip Cookies
```
Recipe: Chocolate Chip Cookies
Ingredients: 2 1/2 c. flour, 3/4 c. sugar, 1 tsp. baking powder, 1 tsp. salt,
1 1/2 c. cold unsalted butter, 1 c. sugar, 3 eggs, 1 c. buttermilk,
1 c. semi-sweet chocolate chips
Instructions:
1. Mix flour, sugar and salt.
2. Add egg, margarine and vanilla.
3. Stir in beaten eggs and chocolate chips.
4. Drop onto ungreased cookie sheet.
5. Bake at 325 degrees for 10 minutes or until lightly browned.
6. Remove from oven and let cool.
7. Cut into small pieces.
<|endofrecipe|>
```
✅ Correct structure, realistic temperature and bake time, logical step ordering.

### Prompt 2: Chicken Soup
```
Recipe: Chicken Soup
Ingredients: 1/2 medium onion (chopped), 1 medium zucchini (grated),
diced green chilies, 1 lb. ground beef, 2 cans cream-style corn,
1 can cream of mushroom soup, 2 small cans chicken broth
Instructions:
1. Combine water, bouillon cubes and tomato sauce.
2. Beat until smooth; add remaining ingredients.
3. Make meatballs.
4. Add enough broth to moisten with butter or margarine.
5. Cook 10 minutes.
6. Add to skillet; stir.
<|endofrecipe|>
```
✅ Correct template structure and grammatically fluent steps.
⚠️ Ingredient hallucination present (zucchini, ground beef in chicken soup) — expected at this loss level.

---

## ✅ Strengths & ⚠️ Limitations

**Strengths**
- Consistent `Recipe → Ingredients → Instructions → <|endofrecipe|>` format every generation
- Grammatically fluent, well-formed instruction steps
- Realistic cooking quantities, temperatures, and procedural language
- Stable sequence termination

**Limitations**
- Occasional ingredient hallucination (off-topic ingredients selected)
- Imperfect consistency between ingredient list and later instructions
- Weak long-range semantic dependency in longer generations (128 token context limit)

---

## 🚀 Future Improvements

- **Longer context window** (512–1024 tokens) to retain ingredients through full generation
- **Cuisine conditioning** tags (`Cuisine: Indian`) to reduce hallucination

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

## 📁 Repository Structure

```
TinyChef/
├── SLM_Recipe_Generator.ipynb   # Full training notebook (Kaggle-ready)
├── best_model_params.pt         # Trained model weights
├── README.md                    # This file
└── LICENSE                      # MIT License
```

---

## 📚 References

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) — Eldan & Li, 2023
- [RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation](https://recipenlg.cs.put.poznan.pl/) — Bień et al., ACL INLG 2020
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
