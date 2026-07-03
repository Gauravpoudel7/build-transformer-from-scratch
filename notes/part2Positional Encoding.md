# Transformer Input Representation

## 1. Maximum Sequence Length

```python
max_sequence_length = 10
```

This tells the model:

> **The longest sentence the model can process contains 10 tokens.**

If a sentence contains **more than 10 tokens**, the model cannot process it directly.

### Option 1: Truncation

* Keep only the **first 10 tokens**.
* Discard the remaining tokens.

### Option 2: Padding

If a sentence contains **fewer than 10 tokens**, add special **`<PAD>`** tokens until the sequence length becomes 10.

Example:

```text
Original:
I love AI

After Padding:
I love AI <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
```

---

## 2. Embedding Dimension (`d_model`)

```python
d_model = 512
```

This is called the **embedding dimension** (or **hidden dimension**).

It means:

* Every **token** is represented by **512 numbers**.
* These 512 values capture the semantic meaning of the token.

### Why 512?

The original Transformer paper used:

```python
d_model = 512
```

because it provided a good balance between:

* Enough capacity to learn complex relationships.
* Reasonable computational cost.

There is **nothing special about 512**. Different models use different embedding dimensions depending on their size and purpose.

---

## 3. Token Shape

A **single token** (for example, `"I"`) is represented as:

```text
(512,)
```

or simply:

```text
512-dimensional vector
```

If a sentence contains **10 tokens**, the entire embedding matrix becomes:

```text
(10, 512)
```

where:

* **10** → Number of tokens (sequence length)
* **512** → Embedding dimension

---

## 4. Positional Encoding

Since Transformers process all tokens simultaneously, they need information about the **position** of each token.

For a maximum sequence length of 10, we create positional vectors of shape:

```text
(10, 512)
```

However, we **do not always use all 10 positional vectors**.

Instead, we select only the positional vectors needed for the current sentence.

Examples:

* Sentence with **3 tokens** → Use the first **3** positional vectors.
* Sentence with **5 tokens** → Use the first **5** positional vectors.
* Sentence with **10 tokens** → Use all **10** positional vectors.

In code:

```python
position_vectors = positional_encoding[:sequence_length]
```

The positional vectors must have the **same shape** as the embedded tokens before they can be added.

Example:

```text
Embedded Tokens (EV):      (3, 512)
Positional Vectors (PV):   (3, 512)

Transformer Input = EV + PV
```

**Rule:**

> The number of positional vectors must always match the number of embedded tokens.
