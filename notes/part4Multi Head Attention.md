# Transformer From Scratch - Part 4: Multi-Head Attention

> **Paper:** *Attention Is All You Need*  
> **Section:** 3.2.2

---

# Why is Multi-Head Attention Needed?

A **single attention head** can only focus well on **one type of relationship** at a time.

For example, consider the sentence:

> **"The small dog chased the cat yesterday."**

One attention head might learn:

- **dog → chased** (Who performed the action?)

But it may miss other important relationships such as:

- **small → dog** (Description)
- **cat → chased** (Who received the action?)
- **yesterday → chased** (When did it happen?)

A single head cannot easily capture every relationship simultaneously.

---

# Multi-Head Attention

Instead of using **one attention head**, the Transformer uses **multiple attention heads**.

Each head learns a different relationship.

For example:

| Head | Learns |
|------|---------|
| Head 1 | Who performed the action? |
| Head 2 | Who received the action? |
| Head 3 | Descriptions (adjectives) |
| Head 4 | Time information |
| Head 5 | Location information |
| Head 6 | Object relationships |
| Head 7 | Long-distance dependencies |
| Head 8 | General sentence context |

Finally, the outputs of all heads are combined into one output.

---

# Simple Analogy

Imagine reading a book using **8 different highlighters**.

- 🟡 One highlights names.
- 🟢 One highlights dates.
- 🔵 One highlights places.
- 🟣 One highlights actions.
- 🟠 One highlights descriptions.
- 🔴 One highlights important events.
- ⚫ One highlights relationships.
- ⚪ One highlights keywords.

Using all eight highlighters gives you a much better understanding of the story than using only one.

This is exactly what **Multi-Head Attention** does.

---

# Another Way to Think About It

Imagine **8 different people** reading the same sentence.

Each person focuses on something different.

After everyone finishes reading, they discuss what they found.

The final understanding is much richer than what any one person could have understood alone.

---

# Code Explanation

---

# Step 1: Define the Model

```python
sequence_length = 3

d_model = 512

num_heads = 8

dk = d_model // num_heads
```

Suppose our sentence is:

```
I love Transformers
```

There are:

- **3 words**
- Each word is represented using **512 numbers**

The input matrix looks like this:

| Word | Vector |
|------|--------|
| I | 512 numbers |
| love | 512 numbers |
| Transformers | 512 numbers |

Shape:

```
(3,512)
```

---

# Step 2: Input Embeddings

```python
X = np.random.randn(sequence_length, d_model)
```

Normally the input comes from:

```
Embedding Layer
        ↓
Positional Encoding
        ↓
Input Matrix (X)
```

Since we haven't built the complete Transformer yet, we create random embeddings.

Example:

```
I

[0.2 0.8 ... 512 values]

love

[-0.1 1.2 ... 512 values]

Transformers

[0.9 -0.4 ... 512 values]
```

Shape:

```
(3,512)
```

---

# Step 3: Create Weight Matrices

```python
WQ
WK
WV
WO
```

These are trainable matrices.

Think of them as four different filters.

They learn:

- How to create **Queries**
- How to create **Keys**
- How to create **Values**
- How to combine the final outputs

Each matrix has shape:

```
(512,512)
```

---

# Step 4: Generate Q, K and V

```python
Q = X @ WQ

K = X @ WK

V = X @ WV
```

Every word is multiplied by the corresponding weight matrix.

Example:

```
Input

I

↓

512 numbers

↓

Multiply by WQ

↓

Query Vector
```

The same happens for every word.

Result:

```
Q

(3,512)

K

(3,512)

V

(3,512)
```

---

## Why are they still 512?

Because:

```
WQ

(512,512)
```

Multiplying

```
(3,512)

×

(512,512)

↓

(3,512)
```

The reduction to **64 dimensions** happens after splitting into heads.

---

# Step 5: Split into Multiple Heads

```python
Q.reshape(3,8,64)
```

This is the most important step.

Originally each word has:

```
512 numbers
```

Split those 512 numbers into **8 equal parts**.

```
512

↓

64

64

64

64

64

64

64

64
```

Now each word contains **8 smaller vectors**.

Shape becomes:

```
(3,8,64)
```

Meaning:

- 3 words
- 8 heads
- 64 numbers per head

---

# Step 6: Transpose

```python
Q = np.transpose(Q,(1,0,2))
```

Before transpose:

```
(3,8,64)

↓

Words

↓

Heads
```

After transpose:

```
(8,3,64)

↓

Heads

↓

Words
```

Now each head has access to all words independently.

For example:

### Head 1

```
I

love

Transformers
```

### Head 2

```
I

love

Transformers
```

...

### Head 8

```
I

love

Transformers
```

Now every head performs its own attention calculation.

---

# Step 7: Softmax

```python
softmax()
```

Suppose attention scores are:

```
8

5

2
```

Softmax converts them into probabilities:

```
0.94

0.05

0.01
```

Every row always sums to:

```
1
```

---

# Step 8: Each Head Performs Attention

The loop:

```python
for head in range(8):
```

runs the attention calculation for every head.

Each head computes:

```
Q

×

Kᵀ
```

Example attention scores:

```
             I    love   Transformers

I            7      4         1

love         5      8         3

Transformers 2      6         9
```

Next:

Divide by:

```
√64 = 8
```

Then apply Softmax.

Example:

```
0.70

0.20

0.10
```

Finally multiply by:

```
V
```

Each head produces:

```
(3,64)
```

This process repeats for all 8 heads.

Result:

```
Head 1

(3,64)

Head 2

(3,64)

...

Head 8

(3,64)
```

---

# Step 9: Stack All Heads

```python
np.stack()
```

Initially:

```
[
Head1

Head2

...

Head8
]
```

After stacking:

```
(8,3,64)
```

---

# Step 10: Concatenate Heads

Now combine all heads.

```
Head1

64 values

Head2

64 values

...

Head8

64 values
```

Concatenate them:

```
64

+

64

+

64

+

64

+

64

+

64

+

64

+

64

=

512
```

Now each word once again has:

```
512 numbers
```

Shape:

```
(3,512)
```

Exactly the same size as the original embedding.

---

# Step 11: Final Linear Projection

```python
output = concat @ WO
```

The paper does not use the concatenated vectors directly.

Instead, they are multiplied by another trainable matrix:

```
WO
```

This matrix combines information learned by all attention heads.

Output shape remains:

```
(3,512)
```

---

# Final Result

Each word started as:

```
512 numbers
```

After Multi-Head Attention, each word is **still represented by 512 numbers**, but those numbers now contain information gathered from **all other words** in the sentence.

For example, the word:

```
Transformers
```

now contains information from:

- I
- love
- Transformers

weighted according to how much attention each head gave to those words.

---

# Complete Flow

```
Sentence

"I love Transformers"

        │

        ▼

Embedding + Positional Encoding

        │

        ▼

(3,512)

        │

        ▼

Generate Q, K, V

        │

        ▼

Q (3,512)
K (3,512)
V (3,512)

        │

        ▼

Split into 8 Heads

        │

        ▼

(8,3,64)

        │

        ▼

Each Head Performs

QKᵀ

↓

Scale

↓

Softmax

↓

Weights × V

        │

        ▼

8 Outputs

(8,3,64)

        │

        ▼

Concatenate

        │

        ▼

(3,512)

        │

        ▼

Final Linear Layer (WO)

        │

        ▼

Final Output

(3,512)
```

---

# Key Idea to Remember

A **single attention head** can only learn **one type of relationship** between words.

By using **8 attention heads**, the Transformer allows eight independent attention mechanisms to analyze the same sentence from different perspectives at the same time.

Finally, the outputs of all heads are combined into one rich representation, giving the model a much better understanding of the sentence than a single attention head could achieve.