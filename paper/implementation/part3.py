"""
============================================================
Transformer From Scratch - Part 3
Scaled Dot-Product Attention
============================================================

Paper:
Attention Is All You Need
Section 3.2.1

This program implements the complete Scaled Dot-Product
Attention mechanism.

Pipeline

Input Embeddings
        │
        ▼
Generate Q, K, V
        │
        ▼
Q × Kᵀ
        │
        ▼
Divide by √dk
        │
        ▼
(Optional) Apply Mask
        │
        ▼
Softmax
        │
        ▼
Multiply by V
        │
        ▼
Attention Output
"""

import numpy as np
import math

# ============================================================
# Step 1 : Model Parameters
# ============================================================

"""
Example sentence

"I love Transformers"

Number of words = 3

Embedding dimension = 512

For one attention head

dk = dv = 64
"""

sequence_length = 3
d_model = 512
dk = 64
dv = 64


# ============================================================
# Step 2 : Create Example Input Embeddings
# ============================================================

"""
Normally these embeddings come from

Embedding Layer
+
Positional Encoding

Shape

(sequence_length, d_model)

(3,512)
"""

np.random.seed(42)

X = np.random.randn(sequence_length, d_model)

print("Input Embedding Shape")
print(X.shape)


# ============================================================
# Step 3 : Create Weight Matrices
# ============================================================

"""
Attention learns three matrices

WQ
WK
WV

Each transforms

512 dimensions

↓

64 dimensions
"""

WQ = np.random.randn(d_model, dk)
WK = np.random.randn(d_model, dk)
WV = np.random.randn(d_model, dv)


# ============================================================
# Step 4 : Compute Queries, Keys and Values
# ============================================================

"""
Q = XWQ

K = XWK

V = XWV
"""

Q = np.matmul(X, WQ)

K = np.matmul(X, WK)

V = np.matmul(X, WV)

print("\nQ Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)


# ============================================================
# Step 5 : Compute Attention Scores
# ============================================================

"""
Attention Scores

QKᵀ

Shapes

Q

(3,64)

Kᵀ

(64,3)

Result

(3,3)

Every word is compared with
every other word.
"""

scores = np.matmul(Q, K.T)

print("\nAttention Score Shape")
print(scores.shape)

print("\nRaw Attention Scores")
print(scores)


# ============================================================
# Step 6 : Scale Scores
# ============================================================

"""
Paper

scores = scores / √dk

Without scaling

dot products become very large

Softmax becomes unstable.
"""

scores = scores / math.sqrt(dk)

print("\nScaled Scores")
print(scores)


# ============================================================
# Step 7 : Optional Mask
# ============================================================

"""
Masking is mainly used inside
the decoder.

Here we show how masking works.

Mask Matrix

0  -inf -inf
0    0  -inf
0    0    0

Future words receive -infinity.

If you are implementing
encoder self-attention

skip this step.
"""

use_mask = False

if use_mask:

    mask = np.triu(
        np.ones((sequence_length, sequence_length)),
        k=1
    )

    scores = np.where(mask == 1, -1e9, scores)


# ============================================================
# Step 8 : Softmax Function
# ============================================================

"""
Softmax converts

scores

↓

probabilities

Each row sums to 1.
"""

def softmax(x):

    # subtract maximum for numerical stability

    x = x - np.max(x, axis=-1, keepdims=True)

    exp = np.exp(x)

    return exp / np.sum(exp, axis=-1, keepdims=True)


attention_weights = softmax(scores)

print("\nAttention Weights")
print(attention_weights)

print("\nRow Sum")
print(np.sum(attention_weights, axis=1))


# ============================================================
# Step 9 : Compute Final Attention Output
# ============================================================

"""
Output

AttentionWeights × V

Shapes

(3,3)

×

(3,64)

↓

(3,64)
"""

attention_output = np.matmul(attention_weights, V)

print("\nAttention Output Shape")
print(attention_output.shape)


# ============================================================
# Step 10 : Display Results
# ============================================================

print("\nFirst 10 values of first output vector")

print(attention_output[0][:10])

print("\nFirst 10 values of second output vector")

print(attention_output[1][:10])

print("\nFirst 10 values of third output vector")

print(attention_output[2][:10])


# ============================================================
# Step 11 : Complete Summary
# ============================================================

"""
Input

(3,512)

↓

Q

(3,64)

↓

K

(3,64)

↓

V

(3,64)

↓

QKᵀ

(3,3)

↓

Scale

↓

Softmax

↓

Attention Matrix

(3,3)

↓

Multiply by V

↓

Output

(3,64)
"""

print("\nPipeline Completed Successfully!")