"""
============================================================
Transformer From Scratch - Part 4
Multi-Head Attention
============================================================

Paper:
Attention Is All You Need
Section 3.2.2

This program implements Multi-Head Attention
completely from scratch using NumPy.

Pipeline

Input Embedding
        │
        ▼
Generate Q, K, V for Head 1
        │
Generate Q, K, V for Head 2
        │
...
        │
Generate Q, K, V for Head 8
        │
        ▼
Scaled Dot Product Attention
        │
        ▼
8 Attention Outputs
        │
        ▼
Concatenate Heads
        │
        ▼
Final Linear Projection
        │
        ▼
Output
"""

import numpy as np
import math

# ============================================================
# Step 1 : Model Parameters
# ============================================================

"""
Example sentence

"I love Transformers"

Sequence Length = 3

Model Dimension = 512

Heads = 8

Each head learns

64-dimensional attention vectors.

Because

512 / 8 = 64
"""

sequence_length = 3

d_model = 512

num_heads = 8

dk = d_model // num_heads

dv = dk

np.random.seed(42)


# ============================================================
# Step 2 : Input Embedding
# ============================================================

"""
Normally this comes from

Embedding Layer
+
Positional Encoding

Shape

(3,512)
"""

X = np.random.randn(sequence_length, d_model)

print("Input Shape")
print(X.shape)


# ============================================================
# Step 3 : Create Weight Matrices
# ============================================================

"""
Each head has its own

WQ
WK
WV

Instead of creating

24 separate matrices,

we create one large matrix

(512,512)

Later we split it into 8 heads.
"""

WQ = np.random.randn(d_model, d_model)

WK = np.random.randn(d_model, d_model)

WV = np.random.randn(d_model, d_model)

WO = np.random.randn(d_model, d_model)


# ============================================================
# Step 4 : Generate Q K V
# ============================================================

"""
Q = XWQ

K = XWK

V = XWV

Shapes

(3,512)
"""

Q = X @ WQ

K = X @ WK

V = X @ WV

print("\nQ Shape", Q.shape)
print("K Shape", K.shape)
print("V Shape", V.shape)


# ============================================================
# Step 5 : Split into Multiple Heads
# ============================================================

"""
Current Shape

(3,512)

Need

(3,8,64)

Meaning

3 words

8 attention heads

64 dimensions/head
"""

Q = Q.reshape(sequence_length, num_heads, dk)

K = K.reshape(sequence_length, num_heads, dk)

V = V.reshape(sequence_length, num_heads, dv)

print("\nAfter Split")

print(Q.shape)


# ============================================================
# Step 6 : Move Heads Forward
# ============================================================

"""
Transpose

(3,8,64)

↓

(8,3,64)

Now

Each head becomes an independent matrix.
"""

Q = np.transpose(Q, (1, 0, 2))

K = np.transpose(K, (1, 0, 2))

V = np.transpose(V, (1, 0, 2))

print("\nAfter Transpose")

print(Q.shape)


# ============================================================
# Step 7 : Softmax Function
# ============================================================

def softmax(x):

    x = x - np.max(x, axis=-1, keepdims=True)

    exp = np.exp(x)

    return exp / np.sum(exp, axis=-1, keepdims=True)


# ============================================================
# Step 8 : Attention for Every Head
# ============================================================

"""
Each head independently computes

Attention(Q,K,V)

Output

(3,64)
"""

head_outputs = []

for head in range(num_heads):

    q = Q[head]

    k = K[head]

    v = V[head]

    # -----------------------------------------
    # QKᵀ
    # -----------------------------------------

    scores = q @ k.T

    # -----------------------------------------
    # Scale
    # -----------------------------------------

    scores = scores / math.sqrt(dk)

    # -----------------------------------------
    # Softmax
    # -----------------------------------------

    weights = softmax(scores)

    # -----------------------------------------
    # Multiply by V
    # -----------------------------------------

    output = weights @ v

    head_outputs.append(output)

print("\nNumber of Heads")

print(len(head_outputs))


# ============================================================
# Step 9 : Stack Heads
# ============================================================

"""
Current

8 matrices

Each

(3,64)

↓

Stack

(8,3,64)
"""

head_outputs = np.stack(head_outputs)

print("\nStack Shape")

print(head_outputs.shape)


# ============================================================
# Step 10 : Concatenate Heads
# ============================================================

"""
Need

(3,512)

Transpose

(8,3,64)

↓

(3,8,64)

↓

reshape

(3,512)
"""

head_outputs = np.transpose(head_outputs, (1, 0, 2))

concat = head_outputs.reshape(sequence_length, d_model)

print("\nConcatenated Shape")

print(concat.shape)


# ============================================================
# Step 11 : Final Linear Projection
# ============================================================

"""
Paper

Concat(heads)

↓

WO

↓

Final Output
"""

output = concat @ WO

print("\nFinal Output Shape")

print(output.shape)


# ============================================================
# Step 12 : Display Result
# ============================================================

print("\nFirst 10 values of Word 1")

print(output[0][:10])

print("\nFirst 10 values of Word 2")

print(output[1][:10])

print("\nFirst 10 values of Word 3")

print(output[2][:10])


# ============================================================
# Summary
# ============================================================

"""
Input

(3,512)

↓

Generate Q,K,V

↓

Split into 8 Heads

↓

Each Head

(3,64)

↓

Scaled Dot Product Attention

↓

8 Outputs

↓

Concatenate

(3,512)

↓

Linear Projection

↓

Final Output

(3,512)
"""

print("\nMulti-Head Attention Completed Successfully!")