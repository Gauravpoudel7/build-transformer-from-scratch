# Transformer From Scratch – Part 3: Scaled Dot-Product Attention

## Paper
**Attention Is All You Need (Section 3.2.1)**

This document explains the complete implementation of **Scaled Dot-Product Attention** from the Transformer paper in simple English.

---

# Big Picture

Imagine the sentence:

> **"I love Transformers"**

There are **3 words**:

- I
- love
- Transformers

The goal of attention is for every word to ask:

> **Which other words should I pay attention to?**

For example:

- **Transformers** should pay attention to **love**, because it tells us something about Transformers.
- **love** should pay attention to **I**, because "I" is performing the action.

This process is called **Attention**.

---

# Step 1 – Model Parameters

```python
sequence_length = 3
d_model = 512
dk = 64
dv = 64
```

- `sequence_length = 3` because there are three words.
- `d_model = 512` means every word is represented by **512 numbers**.
- `dk = 64` and `dv = 64` are the dimensions used by one attention head.

Instead of storing the word "I" as text, the model stores it as:

```text
[0.41, -0.73, 0.15, ..., 512 values]
```

---

# Step 2 – Input Embeddings

```python
X = np.random.randn(sequence_length, d_model)
```

Shape:

```
(3, 512)
```

Meaning:

- 3 rows → 3 words
- 512 columns → 512 features for each word

In a real Transformer, this comes from:

- Embedding Layer
- Positional Encoding

---

# Step 3 – Weight Matrices

The Transformer learns three matrices:

- WQ
- WK
- WV

Each has shape:

```
(512, 64)
```

These convert a 512-dimensional embedding into a 64-dimensional representation for one attention head.

---

# Step 4 – Compute Queries, Keys, and Values

```python
Q = X @ WQ
K = X @ WK
V = X @ WV
```

Shapes:

- Q → (3,64)
- K → (3,64)
- V → (3,64)

## What are Q, K, and V?

Think of a classroom:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What information do I have?"
- **Value (V):** "What information will I provide?"

Every word creates one Query, one Key, and one Value.

---

# Step 5 – Compute Attention Scores

```python
scores = Q @ K.T
```

Shapes:

```
Q      (3,64)
K.T    (64,3)
----------------
Result (3,3)
```

Every Query is compared with every Key.

Example:

```
[[9,2,1],
 [3,8,2],
 [1,5,7]]
```

Each row shows how much one word attends to every other word.

---

# Why Dot Product?

The dot product measures similarity.

Example:

```
A = [1,2]
B = [2,4]

Dot Product = 1×2 + 2×4 = 10
```

Higher values mean vectors are more similar.

---

# Step 6 – Scale Scores

```python
scores = scores / math.sqrt(dk)
```

Since:

```
dk = 64
√64 = 8
```

Scaling prevents very large values that would make Softmax unstable.

---

# Step 7 – Optional Mask

Decoder self-attention uses masking.

Example mask:

```
0   -∞   -∞
0    0   -∞
0    0    0
```

This prevents a word from seeing future words.

Your code disables masking:

```python
use_mask = False
```

because encoder attention does not need it.

---

# Step 8 – Softmax

Softmax converts scores into probabilities.

Example:

```
[2,1,0]
```

becomes approximately:

```
[0.66,0.24,0.10]
```

Every row sums to **1**.

For numerical stability:

```python
x = x - np.max(x)
```

This prevents overflow when computing exponentials.

---

# Step 9 – Compute Attention Output

```python
attention_output = attention_weights @ V
```

Shapes:

```
(3,3)
×
(3,64)
=
(3,64)
```

Example:

Weights:

```
[0.7,0.2,0.1]
```

Output:

```
0.7 × Word1
+0.2 × Word2
+0.1 × Word3
```

The model combines information from all words according to their importance.

---

# Why is the Output Still (3,64)?

The number of words does not change.

Instead, each word gets a **new representation** that now contains information from other relevant words.

Input:

```
3 words × 64 values
```

Output:

```
3 updated words × 64 values
```

---

# Complete Pipeline

```
Sentence
"I love Transformers"

        │
        ▼
Embedding Layer
(3,512)

        │
        ▼
Positional Encoding
(3,512)

        │
        ▼
Q = XWQ
K = XWK
V = XWV

        │
        ▼
QKᵀ
(3,3)

        │
        ▼
Divide by √64

        │
        ▼
(Optional) Mask

        │
        ▼
Softmax

        │
        ▼
Attention Weights

        │
        ▼
Attention Weights × V

        │
        ▼
Attention Output
(3,64)
```

# Mapping to the Research Paper

| Paper | Code |
|-------|------|
| Input embeddings | `X` |
| WQ | `WQ` |
| WK | `WK` |
| WV | `WV` |
| Q = XWQ | `Q = X @ WQ` |
| K = XWK | `K = X @ WK` |
| V = XWV | `V = X @ WV` |
| QKᵀ | `scores = Q @ K.T` |
| Divide by √dk | `scores /= math.sqrt(dk)` |
| Mask | `np.where(...)` |
| Softmax | `attention_weights = softmax(scores)` |
| Multiply by V | `attention_output = attention_weights @ V` |

The implementation follows the paper's formula:

**Attention(Q,K,V) = softmax(QKᵀ / √dk) V**

---

# What Comes Next?

After Scaled Dot-Product Attention, the Transformer continues with:

1. Multi-Head Attention
2. Residual Connection + Layer Normalization
3. Position-wise Feed-Forward Network
4. Residual Connection + Layer Normalization
5. Stack multiple encoder/decoder layers

Scaled Dot-Product Attention is the core building block that enables Transformers to understand relationships between words.
