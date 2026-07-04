# Transformer From Scratch - Understanding Matrix Shapes

## The One Rule to Remember

Every matrix multiplication follows:

``` text
(m × n) × (n × p) = (m × p)
```

The **inside dimensions must match**. The **outside dimensions become
the result**.

Examples:

-   `(3 × 512) × (512 × 64) = (3 × 64)`
-   `(3 × 64) × (64 × 3) = (3 × 3)`
-   `(3 × 3) × (3 × 64) = (3 × 64)`

------------------------------------------------------------------------

# Step 1: Our Sentence

Sentence:

``` text
"I love Transformers"
```

There are **3 words (tokens)**.

``` text
Token 1 = I
Token 2 = love
Token 3 = Transformers
```

So:

``` text
Sequence Length = 3
```

------------------------------------------------------------------------

# Step 2: Each Word Becomes a Vector

Each word is converted into a vector of **512 numbers**.

``` text
Token 1 → 512 values
Token 2 → 512 values
Token 3 → 512 values
```

Stacking them forms the input matrix:

``` text
X

[
 Token1 (512 values)
 Token2 (512 values)
 Token3 (512 values)
]
```

Shape:

``` text
(3, 512)
```

-   Rows = number of tokens
-   Columns = embedding dimension

------------------------------------------------------------------------

# Step 3: Why WQ is (512, 64)

The weight matrix projects each 512-dimensional embedding into a
64-dimensional space.

``` text
WQ = (512, 64)
```

Think of it as:

``` text
512 features
      ↓
64 features
```

------------------------------------------------------------------------

# Step 4: Computing Q

``` text
Q = X × WQ
```

Shapes:

``` text
X  = (3, 512)
WQ = (512, 64)
```

Using the rule:

``` text
(3 × 512) × (512 × 64)
```

Result:

``` text
Q = (3, 64)
```

The same applies to:

``` text
K = (3, 64)
V = (3, 64)
```

------------------------------------------------------------------------

# Step 5: Why Transpose K?

Originally:

``` text
K = (3, 64)
```

Transpose swaps rows and columns:

``` text
Kᵀ = (64, 3)
```

------------------------------------------------------------------------

# Step 6: Computing Attention Scores

``` text
Scores = Q × Kᵀ
```

Shapes:

``` text
Q   = (3, 64)
Kᵀ  = (64, 3)
```

Multiplication:

``` text
(3 × 64) × (64 × 3)
```

Result:

``` text
Scores = (3, 3)
```

## Why (3,3)?

Every token compares with every other token.

  Asking Token   Looks At
  -------------- -----------------------
  I              I, Love, Transformers
  Love           I, Love, Transformers
  Transformers   I, Love, Transformers

So the attention score matrix contains:

``` text
        I   Love  Transformers

I        ?     ?       ?

Love     ?     ?       ?

Transformers ?  ?       ?
```

If there were 10 words instead of 3, the score matrix would be:

``` text
(10, 10)
```

------------------------------------------------------------------------

# Step 7: Softmax

Softmax changes the values, **not the shape**.

Input:

``` text
(3, 3)
```

Output:

``` text
(3, 3)
```

Each row becomes probabilities that sum to 1.

------------------------------------------------------------------------

# Step 8: Final Attention Output

``` text
Attention Output = Attention Weights × V
```

Shapes:

``` text
Attention Weights = (3, 3)
V                 = (3, 64)
```

Multiplication:

``` text
(3 × 3) × (3 × 64)
```

Result:

``` text
Output = (3, 64)
```

The attention matrix contains **weights**, while `V` contains the actual
information.

For example:

``` text
Weights = [0.7, 0.2, 0.1]

Output₁ =
0.7 × V₁ +
0.2 × V₂ +
0.1 × V₃
```

Since every `V` has 64 values, the output also has 64 values.

------------------------------------------------------------------------

# Complete Shape Flow

``` text
Sentence

"I love Transformers"

↓

3 words

↓

Embedding

(3, 512)

↓

Q = (3, 64)
K = (3, 64)
V = (3, 64)

↓

Q × Kᵀ

↓

(3, 64) × (64, 3)

↓

Scores = (3, 3)

↓

Softmax

↓

Attention Weights = (3, 3)

↓

Attention Weights × V

↓

(3, 3) × (3, 64)

↓

Output = (3, 64)
```

# Quick Summary

-   **(3, 512)** → 3 tokens, each with 512 features.
-   **(3, 64)** → each token projected into a 64-dimensional attention
    space.
-   **(3, 3)** → every token compared with every other token.
-   **(3, 64)** → new representation for each token after attention.
