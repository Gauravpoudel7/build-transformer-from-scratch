# Transformer From Scratch - Part 7
# Encoder Layer

## Paper

**Attention Is All You Need**

**Section 3.1**

---

# What is an Encoder Layer?

An Encoder Layer is the main building block of the Transformer.

Instead of using only one attention calculation, the Transformer combines several operations together to better understand the input sentence.

One Encoder Layer contains four main steps:

1. Multi-Head Attention
2. Residual Connection + Layer Normalization
3. Feed Forward Network
4. Residual Connection + Layer Normalization

---

# Encoder Layer Pipeline

```text
Input
   │
   ▼
Multi-Head Attention
   │
   ▼
Residual Connection
   │
   ▼
Layer Normalization
   │
   ▼
Feed Forward Network
   │
   ▼
Residual Connection
   │
   ▼
Layer Normalization
   │
   ▼
Encoder Output
```

---

# Example Sentence

Suppose our sentence is:

```
I love Transformers
```

There are **3 words**.

Each word is represented by **512 numbers**.

Input Shape

```
(3,512)
```

Meaning

```
3 words

↓

512 features for each word
```

---

# Step 1 : Multi-Head Attention

The first operation is Multi-Head Attention.

Each attention head looks at the sentence from a different perspective.

For example

Head 1

```
Who performed the action?
```

Head 2

```
Who received the action?
```

Head 3

```
Descriptions
```

Head 4

```
Time information
```

...

Head 8

```
Another relationship
```

All eight heads work at the same time.

Their outputs are combined into one vector.

Output Shape

```
(3,512)
```

---

# Step 2 : Residual Connection

Instead of keeping only the attention output,

the Transformer also keeps the original input.

Equation

```
Residual = Input + Attention Output
```

This helps preserve important information from the input.

Shape

```
(3,512)

+

(3,512)

↓

(3,512)
```

---

# Why use Residual Connection?

Without residual connections,

information from earlier layers may slowly disappear.

Residual connections allow the model to keep the original information while also learning new information.

Think of it like this:

Instead of replacing your old notes,

you write new notes on top of them.

This way you never lose the important information.

---

# Step 3 : Layer Normalization

After adding the residual connection,

the values can become very large or very small.

Layer Normalization keeps the values balanced.

For every word,

it computes

Mean

↓

Variance

↓

Normalize

Equation

```
Output = (x − mean) / √(variance + ε)
```

Finally,

two learnable parameters are applied.

```
Gamma (γ)

↓

Scale the values

Beta (β)

↓

Shift the values
```

Output Shape

```
(3,512)
```

---

# Why use Layer Normalization?

Layer Normalization makes training more stable.

It keeps the numbers in a reasonable range,

making it easier for the model to learn.

---

# Step 4 : Feed Forward Network

Now each word is processed independently.

Unlike attention,

the Feed Forward Network does **not** compare words.

Instead,

it improves the representation of each word individually.

Pipeline

```
512

↓

2048

↓

ReLU

↓

512
```

Equation

```
FFN(x)

=

max(0, xW1 + b1)

↓

W2 + b2
```

Output Shape

```
(3,512)
```

---

# Why expand to 2048?

The hidden layer is larger than the input.

This gives the network more capacity to learn complex patterns.

Think of it as giving the model a larger workspace before compressing the information back to 512 dimensions.

---

# Step 5 : Second Residual Connection

Again,

the original input to the Feed Forward Network is added back.

Equation

```
Residual

=

Input

+

FFN Output
```

Shape

```
(3,512)

+

(3,512)

↓

(3,512)
```

---

# Step 6 : Second Layer Normalization

The output is normalized one more time.

This produces the final output of the Encoder Layer.

Output Shape

```
(3,512)
```

---

# Complete Flow

```text
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

Multi-Head Attention

        │

        ▼

Residual Connection

        │

        ▼

Layer Normalization

        │

        ▼

Feed Forward Network

        │

        ▼

Residual Connection

        │

        ▼

Layer Normalization

        │

        ▼

Encoder Output

(3,512)
```

---

# Shape Summary

| Step | Shape |
|------|-------|
| Input | (3,512) |
| Multi-Head Attention | (3,512) |
| Residual Connection | (3,512) |
| Layer Normalization | (3,512) |
| Feed Forward Network | (3,512) |
| Residual Connection | (3,512) |
| Layer Normalization | (3,512) |
| Encoder Output | (3,512) |

Notice that the shape never changes.

Only the information inside the vectors becomes richer after every operation.

---

# Simple Analogy

Imagine a student reading a book.

**Multi-Head Attention**

The student looks at the relationships between different words and sentences.

↓

**Residual Connection**

The student keeps their original notes while adding new observations.

↓

**Layer Normalization**

The student organizes the notes so they are neat and easy to understand.

↓

**Feed Forward Network**

The student studies each topic individually and gains a deeper understanding.

↓

**Residual Connection**

The student combines the old knowledge with the new understanding.

↓

**Layer Normalization**

The student organizes everything again before moving to the next chapter.

---

# Key Idea

An Encoder Layer takes an input sentence and makes each word representation more meaningful by:

- Learning relationships between words using Multi-Head Attention.
- Preserving important information with Residual Connections.
- Stabilizing training with Layer Normalization.
- Improving each word independently using a Feed Forward Network.

The output of one Encoder Layer becomes the input to the next Encoder Layer in the Encoder Stack.