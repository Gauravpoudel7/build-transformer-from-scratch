# Transformer From Scratch - Part 5.5: Residual Connection + Layer Normalization

> **Paper:** *Attention Is All You Need*  
> **Section:** 3.1

---

# Why do we need Residual Connection and Layer Normalization?

Let's first understand the problem.

Suppose we have a Transformer with **12 encoder layers**.

The data flows like this:

```
Input

↓

Layer 1

↓

Layer 2

↓

Layer 3

↓

...

↓

Layer 12
```

As the network becomes deeper,

two major problems occur:

1. The model starts forgetting the original information.
2. Training becomes unstable because the values become too large or too small.

The Transformer solves these problems using:

- **Residual Connection**
- **Layer Normalization**

These two operations are applied **after every sub-layer** (Multi-Head Attention and Feed Forward Network).

---

# Overall Pipeline

Suppose Multi-Head Attention has already produced an output.

The Transformer performs:

```
Input
      │
      ▼
Multi-Head Attention
      │
      ▼
Residual Addition
(Input + Attention Output)
      │
      ▼
Layer Normalization
      │
      ▼
Output
```

Exactly the same thing happens after the Feed Forward Network.

---

# Step 1 : Model Parameters

```python
sequence_length = 3

d_model = 512

epsilon = 1e-5
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

Three words.

---

```python
d_model = 512
```

Each word is represented by

```
512 numbers
```

Example

```
I

[0.21 -0.51 1.34 ... 512 values]
```

---

```python
epsilon = 1e-5
```

This is a **very small number**.

It prevents division by zero when we normalize.

Suppose every feature had exactly the same value.

Then

```
Variance = 0
```

Without epsilon

```
Divide by √0

↓

Impossible
```

Adding epsilon

```
Variance + 0.00001
```

makes the calculation safe.

---

# Step 2 : Create Example Input

```python
X = np.random.randn(sequence_length, d_model)
```

Normally,

this input comes from

```
Embedding

↓

Positional Encoding

↓

Previous Transformer Layer
```

Since we haven't built the complete encoder yet,

we create random values.

Shape

```
(3,512)
```

Meaning

```
3 words

512 features each
```

---

# Step 3 : Create Sub-layer Output

```python
sublayer_output = np.random.randn(sequence_length,d_model)
```

Imagine this came from

```
Multi-Head Attention
```

or

```
Feed Forward Network
```

Again

Shape

```
(3,512)
```

Notice

Both matrices have exactly the same size.

This is important because they will be added together.

---

# Why must they have the same shape?

Suppose

```
Input

(3,512)
```

and

```
Attention Output

(3,512)
```

Then

```
Input

+

Attention Output

↓

(3,512)
```

works perfectly.

But

```
(3,512)

+

(3,256)
```

is impossible.

So every Transformer sub-layer always returns

```
(Sequence Length,

d_model)
```

which is

```
(3,512)
```

---

# Step 4 : Residual Connection

```python
residual = X + sublayer_output
```

This is one of the most important ideas in deep learning.

Instead of replacing the input,

we **keep the original input** and **add** the new information.

Suppose

Original Input

```
Word

I

↓

[2 5 1]
```

Attention Output

```
[1 3 4]
```

Residual Connection

```
[2 5 1]

+

[1 3 4]

=

[3 8 5]
```

Nothing is lost.

The model now has

- Original information
- New information

combined together.

---

# Why Residual Connection?

Imagine a student learning mathematics.

Instead of forgetting everything they already know,

they simply add new knowledge.

```
Old Knowledge

+

New Knowledge

↓

Better Knowledge
```

Residual Connection does exactly this.

It helps

- preserve information
- train deeper networks
- avoid the vanishing gradient problem

Without residual connections,

very deep Transformers would be much harder to train.

---

# Step 5 : Compute Mean

```python
mean = np.mean(residual,
               axis=1,
               keepdims=True)
```

Now we perform **Layer Normalization**.

LayerNorm works **one word at a time**.

Suppose one word has only

5 features

instead of 512.

```
Word

[2 6 5 9 3]
```

Mean

```
(2+6+5+9+3)

÷5

=

5
```

The real Transformer computes this over

512 features.

So for every word,

we obtain one mean.

Shape

```
(3,1)
```

Meaning

```
Word1 Mean

Word2 Mean

Word3 Mean
```

---

# Step 6 : Compute Variance

```python
variance = np.var(residual,
                  axis=1,
                  keepdims=True)
```

Variance tells us

how spread out the numbers are.

Suppose

```
Word

[2 6 5 9 3]
```

Mean

```
5
```

Difference from mean

```
-3

1

0

4

-2
```

Square them

```
9

1

0

16

4
```

Average

```
6
```

That is the variance.

Again,

one variance is computed for each word.

Shape

```
(3,1)
```

---

# Step 7 : Normalize

```python
normalized = (residual-mean) / np.sqrt(variance+epsilon)
```

This is the LayerNorm equation.

Suppose

Residual

```
[2 6 5 9 3]
```

Mean

```
5
```

Variance

```
6
```

First subtract the mean

```
[-3

1

0

4

-2]
```

Now divide by

```
√6
```

Result

```
[-1.22

0.41

0

1.63

-0.82]
```

Now the values are centered around zero.

LayerNorm makes learning much more stable.

---

# Why normalize?

Suppose one layer outputs

```
[500

900

300]
```

Another outputs

```
[0.02

0.01

0.03]
```

These scales are completely different.

Training becomes unstable.

LayerNorm converts them into a similar scale.

After normalization,

values usually have

```
Mean ≈ 0

Variance ≈ 1
```

This keeps every layer stable.

---

# Step 8 : Gamma and Beta

```python
gamma = np.ones(d_model)

beta = np.zeros(d_model)
```

LayerNorm doesn't stop after normalization.

It learns

```
Gamma (γ)

Beta (β)
```

Gamma scales the values.

Beta shifts the values.

Initially

```
Gamma

=

1
```

which changes nothing.

Beta

```
=

0
```

which also changes nothing.

During training,

the model learns better values.

Then

```python
output = gamma * normalized + beta
```

This is the final LayerNorm output.

Shape

```
(3,512)
```

---

# Step 9 : Display Results

```python
print(output[0][:10])
```

Each word contains

```
512 numbers
```

Printing all 512 numbers would produce a huge output.

So we only print

```
First 10 values
```

for each word.

---

# What Happens Inside the Transformer?

Suppose the sentence is

```
I

love

Transformers
```

The flow is

```
Embedding

↓

Positional Encoding

↓

Multi-Head Attention

↓

Attention Output

(3,512)

↓

Residual Connection

Input

+

Attention Output

↓

Layer Normalization

↓

Feed Forward Network

↓

Residual Connection

↓

Layer Normalization

↓

Encoder Output
```

Notice that **every major sub-layer is followed by**:

1. Residual Connection
2. Layer Normalization

---

# Why are Residual Connection and LayerNorm always together?

Residual Connection helps the model **remember the original information**.

Layer Normalization keeps the values **stable and well-scaled**.

Together they allow the Transformer to train efficiently even with dozens or hundreds of layers.

---

# Complete Flow

```
Input

(3,512)

        │

        ▼

Sub-layer
(Multi-Head Attention or FFN)

        │

        ▼

Sub-layer Output

(3,512)

        │

        ▼

Residual Addition

Input

+

Sub-layer Output

        │

        ▼

Layer Normalization

        │

        ▼

Subtract Mean

        │

        ▼

Divide by √Variance

        │

        ▼

Scale (γ)

        │

        ▼

Shift (β)

        │

        ▼

Final Output

(3,512)
```

---

# Key Ideas to Remember

- **Residual Connection** adds the original input back to the sub-layer output so that the model does not lose previously learned information.
- **Layer Normalization** normalizes each **word (token)** independently across its **512 features**, making training more stable.
- **Gamma (γ)** and **Beta (β)** are trainable parameters that let the model adjust the normalized values if needed.
- The output shape remains **(sequence_length, d_model)**, which in this example is **(3, 512)**. This allows the output to be passed directly to the next Transformer sub-layer.

---

# Summary

| Step | Input Shape | Output Shape |
|------|-------------|--------------|
| Input | (3,512) | (3,512) |
| Sub-layer Output | (3,512) | (3,512) |
| Residual Addition | (3,512) | (3,512) |
| Compute Mean | (3,512) | (3,1) |
| Compute Variance | (3,512) | (3,1) |
| Normalize | (3,512) | (3,512) |
| Scale & Shift (γ, β) | (3,512) | (3,512) |

The combination of **Residual Connection + Layer Normalization** makes the Transformer easier to train, preserves the original information, and keeps the numerical values stable across many layers.