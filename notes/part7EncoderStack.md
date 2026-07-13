# Transformer From Scratch - Part 8
# Encoder Stack

## Paper

**Attention Is All You Need**

**Section 3.1**

---

# What is an Encoder Stack?

In Part 7, we built **one Encoder Layer**.

However, the Transformer does not use only one Encoder Layer.

Instead, it stacks multiple Encoder Layers on top of each other.

In the original Transformer paper,

```
N = 6
```

This means the input sentence passes through **6 Encoder Layers** one after another.

Each layer learns more about the sentence than the previous one.

---

# Why do we need multiple Encoder Layers?

Imagine reading a difficult paragraph.

The first time you read it,

you understand the basic meaning.

The second time,

you notice relationships between ideas.

The third time,

you understand the deeper meaning.

Every time you read it,

your understanding improves.

The Transformer works in the same way.

Each Encoder Layer improves the representation of every word.

---

# Encoder Stack Pipeline

```text
Input
   │
   ▼
Encoder Layer 1
   │
   ▼
Encoder Layer 2
   │
   ▼
Encoder Layer 3
   │
   ▼
Encoder Layer 4
   │
   ▼
Encoder Layer 5
   │
   ▼
Encoder Layer 6
   │
   ▼
Final Encoder Output
```

---

# Example Sentence

Suppose the sentence is

```
I love Transformers
```

There are

```
3 words
```

Each word has

```
512 features
```

Input Shape

```
(3,512)
```

---

# Step 1 : Input

The input comes from

```
Embedding

+

Positional Encoding
```

Shape

```
(3,512)
```

This input is sent to the first Encoder Layer.

---

# Step 2 : Encoder Layer 1

The first Encoder Layer performs

```
Multi-Head Attention

↓

Residual Connection

↓

Layer Normalization

↓

Feed Forward Network

↓

Residual Connection

↓

Layer Normalization
```

Output Shape

```
(3,512)
```

Although the shape stays the same,

the meaning of every word becomes richer.

---

# Step 3 : Encoder Layer 2

The output of Encoder Layer 1

becomes the input of Encoder Layer 2.

```
Encoder Layer 1

↓

Output

↓

Encoder Layer 2
```

Again,

the shape remains

```
(3,512)
```

But the model now understands the sentence even better.

---

# Step 4 : Encoder Layer 3

The output is passed to Encoder Layer 3.

Again,

```
(3,512)
```

Each word now contains even more contextual information.

---

# Step 5 : Encoder Layer 4

The same process happens again.

Nothing changes in the architecture.

Only the learned information becomes richer.

Output Shape

```
(3,512)
```

---

# Step 6 : Encoder Layer 5

The sentence is processed one more time.

The model keeps improving the representation of every word.

Output Shape

```
(3,512)
```

---

# Step 7 : Encoder Layer 6

This is the final Encoder Layer.

After this layer,

every word contains information gathered from the entire sentence.

Output Shape

```
(3,512)
```

---

# Why does the shape never change?

Every Encoder Layer receives

```
(3,512)
```

and also returns

```
(3,512)
```

The number of features stays the same.

Only the values inside those features become more meaningful.

Think of it like improving a document.

The number of pages stays the same,

but every revision makes the content clearer and more accurate.

---

# How does the Encoder Stack work?

Suppose our input is

```
I love Transformers
```

Initially,

the word **Transformers** only represents itself.

After Encoder Layer 1,

it begins to understand its relationship with

```
I

love
```

After Encoder Layer 2,

it learns even more context.

After Encoder Layer 3,

its representation becomes richer.

This continues until Encoder Layer 6.

By the end,

every word contains information about the entire sentence.

---

# Example Flow

```text
"I love Transformers"

↓

Embedding + Positional Encoding

↓

Encoder Layer 1

↓

Encoder Layer 2

↓

Encoder Layer 3

↓

Encoder Layer 4

↓

Encoder Layer 5

↓

Encoder Layer 6

↓

Final Encoder Output
```

---

# Shape Summary

| Step | Shape |
|------|-------|
| Input | (3,512) |
| Encoder Layer 1 | (3,512) |
| Encoder Layer 2 | (3,512) |
| Encoder Layer 3 | (3,512) |
| Encoder Layer 4 | (3,512) |
| Encoder Layer 5 | (3,512) |
| Encoder Layer 6 | (3,512) |
| Final Encoder Output | (3,512) |

Notice that the shape never changes.

Only the quality of the word representations improves.

---

# Simple Analogy

Imagine six teachers teaching the same student.

Teacher 1 explains the basics.

↓

Teacher 2 adds more details.

↓

Teacher 3 corrects misunderstandings.

↓

Teacher 4 gives deeper examples.

↓

Teacher 5 connects different ideas.

↓

Teacher 6 prepares the student for the final exam.

At the end,

the student understands the topic much better than after the first teacher.

The student is the **word embedding**.

Each teacher is an **Encoder Layer**.

The Encoder Stack is simply several Encoder Layers working one after another.

---

# Why does the Transformer use six Encoder Layers?

A single Encoder Layer can learn useful relationships between words.

However,

using multiple Encoder Layers allows the model to gradually build a deeper understanding of the sentence.

Each layer receives the improved output from the previous layer and refines it even further.

This helps the Transformer capture both simple relationships (such as nearby words) and complex relationships (such as long-distance dependencies).

---

# Key Idea

An Encoder Stack is simply a collection of multiple Encoder Layers placed one after another.

Each Encoder Layer receives the output of the previous layer,

improves the representation of every word,

and passes the improved output to the next layer.

The original Transformer paper uses **6 Encoder Layers**, allowing the model to build increasingly rich and contextual representations of the input sentence before passing them to the decoder.