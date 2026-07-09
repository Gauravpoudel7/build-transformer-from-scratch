# Transformer From Scratch - Part 5: Position-wise Feed Forward Network (FFN)

> **Paper:** *Attention Is All You Need*  
> **Section:** 3.3

---

# What is the Feed Forward Network (FFN)?

After **Multi-Head Attention**, every word has already collected information from all the other words in the sentence.

For example:

```
Sentence

I love Transformers
```

After Multi-Head Attention, the model has:

```
I              → 512 numbers

love           → 512 numbers

Transformers   → 512 numbers
```

Shape:

```
(3,512)
```

Now the Transformer asks:

> **"Can I further improve each word's representation?"**

This is the job of the **Feed Forward Network (FFN).**

Unlike Multi-Head Attention, the FFN **does not compare words with each other.**

Instead, it processes **each word independently.**

---

# FFN Formula

The Transformer paper defines the Feed Forward Network as:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

Pipeline:

```
Input

(3,512)

      │

      ▼

Linear Layer

512 → 2048

      │

      ▼

ReLU

      │

      ▼

Linear Layer

2048 → 512

      │

      ▼

Output

(3,512)
```

---

# Step 1 : Model Parameters

```python
sequence_length = 3

d_model = 512

d_ff = 2048
```

Here,

```python
sequence_length = 3
```

means our sentence contains

```
I
love
Transformers
```

which is **3 words**.

---

```python
d_model = 512
```

Every word is represented using

```
512 numbers
```

For example

```
I

[0.34  -1.21   0.98   ... 512 values]
```

---

```python
d_ff = 2048
```

This is the hidden layer size.

Instead of keeping

```
512 numbers
```

the FFN first expands them to

```
2048 numbers
```

Later it compresses them back to

```
512 numbers
```

Pipeline:

```
512

↓

2048

↓

512
```

---

# Step 2 : Create Example Input

```python
X = np.random.randn(sequence_length, d_model)
```

Normally, this input comes from:

```
Embedding

↓

Positional Encoding

↓

Multi-Head Attention

↓

Feed Forward Network
```

Since we haven't built the complete Transformer yet, we generate random numbers.

Example:

```
Word

I

↓

[0.3 0.9 -0.5 ....]
```

Every word has

```
512 values
```

Overall matrix:

```
(3,512)
```

Meaning:

- 3 words
- 512 features per word

---

# Step 3 : Create Trainable Parameters

```python
W1

b1

W2

b2
```

These are **learnable parameters**.

The model updates them during training.

---

## First Linear Layer

```python
W1 = np.random.randn(512,2048)
```

This matrix converts

```
512 features

↓

2048 features
```

Think of it like expanding a small image into a larger one so you can see more details.

Bias

```python
b1
```

is added after multiplication.

Equation:

```
XW₁ + b₁
```

---

## Second Linear Layer

```python
W2 = np.random.randn(2048,512)
```

This compresses

```
2048 features

↓

512 features
```

Bias

```python
b2
```

is also added.

---

# Step 4 : First Linear Layer

```python
hidden = X @ W1 + b1
```

This is matrix multiplication.

Input:

```
(3,512)
```

Multiply by:

```
(512,2048)
```

Result:

```
(3,2048)
```

Because:

```
(3,512)

×

(512,2048)

↓

(3,2048)
```

Now every word has

```
2048 numbers
```

instead of

```
512 numbers.
```

Example:

Before

```
I

↓

512 numbers
```

After

```
I

↓

2048 numbers
```

Nothing is removed.

The information is simply expanded into a larger feature space.

---

# Why Expand to 2048?

Imagine you're writing an essay.

Instead of summarizing immediately,

you first expand your ideas.

Example:

```
Sentence

↓

Paragraph

↓

Sentence
```

The middle paragraph allows you to express much more information.

The Feed Forward Network works similarly.

```
512

↓

2048

↓

512
```

It gives the model more capacity to learn complex patterns.

---

# Step 5 : ReLU Activation

```python
hidden = np.maximum(0, hidden)
```

ReLU means

```
Negative numbers

↓

0

Positive numbers

↓

Remain unchanged
```

Example:

Before

```
[-8

3

-2

10

5]
```

After

```
[0

3

0

10

5]
```

Only negative values become zero.

---

## Why Use ReLU?

Without ReLU,

the network would be completely linear.

Multiple linear layers are mathematically equivalent to one large linear layer.

ReLU introduces **non-linearity**, allowing the Transformer to learn much more complex patterns.

---

# Step 6 : Second Linear Layer

```python
output = hidden @ W2 + b2
```

Current shape:

```
(3,2048)
```

Multiply by:

```
(2048,512)
```

Result:

```
(3,512)
```

Every word returns to

```
512 numbers.
```

---

# Step 7 : Display Results

```python
print(output[0][:10])
```

This prints

only

```
first 10 values
```

of the first word.

Remember,

every word actually contains

```
512 numbers.
```

Printing all 512 numbers would make the output too large.

---

# What Does the FFN Actually Do?

Suppose after Multi-Head Attention we have

```
Word

I

↓

512 numbers
```

The Feed Forward Network processes it like this:

```
512

↓

Linear Layer

↓

2048

↓

ReLU

↓

2048

↓

Linear Layer

↓

512
```

The same happens independently for every word.

---

# Why is it Called "Position-wise"?

Because **each position (each word)** is processed separately.

Suppose the sentence is

```
I

love

Transformers
```

The FFN performs:

```
Word 1

512

↓

2048

↓

512

-----------------

Word 2

512

↓

2048

↓

512

-----------------

Word 3

512

↓

2048

↓

512
```

Notice:

- Word **I** is processed alone.
- Word **love** is processed alone.
- Word **Transformers** is processed alone.

Unlike Multi-Head Attention,

the FFN **does not mix information between different words.**

---

# Complete Flow

```
Sentence

"I love Transformers"

        │

        ▼

Embedding

        │

        ▼

Positional Encoding

        │

        ▼

Multi-Head Attention

        │

        ▼

Output

(3,512)

        │

        ▼

First Linear Layer

512 → 2048

        │

        ▼

ReLU

        │

        ▼

Second Linear Layer

2048 → 512

        │

        ▼

Final Output

(3,512)
```

---

# Key Idea to Remember

- **Multi-Head Attention** allows each word to gather information from **other words** in the sentence.
- **Feed Forward Network (FFN)** then **improves each word's own representation independently**.
- The FFN first **expands** each 512-dimensional vector to **2048 dimensions**, applies **ReLU** to learn complex patterns, and then **compresses** it back to **512 dimensions**.
- The output shape remains **(sequence_length, 512)** so it can be passed directly to the next Transformer layer.

---

# Summary

| Step | Input Shape | Output Shape |
|------|------------|--------------|
| Multi-Head Attention Output | (3,512) | (3,512) |
| Linear Layer 1 | (3,512) | (3,2048) |
| ReLU | (3,2048) | (3,2048) |
| Linear Layer 2 | (3,2048) | (3,512) |

The Feed Forward Network does **not** allow words to communicate with each other. It simply improves each word's feature representation individually before passing it to the next Transformer layer.