"""
============================================================
Transformer From Scratch - Part 6
Complete Encoder Layer
(Part 1)
============================================================

Paper:
Attention Is All You Need

Section 3.1

This program builds one complete
Transformer Encoder Layer.

Encoder Layer

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

Number of Heads = 8

Feed Forward Hidden Size = 2048
"""

sequence_length = 3

d_model = 512

num_heads = 8

dk = d_model // num_heads

dv = dk

d_ff = 2048

epsilon = 1e-5

np.random.seed(42)

# ============================================================
# Step 2 : Example Input
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
# Step 3 : Create Multi-Head Attention Parameters
# ============================================================

"""
Trainable matrices

WQ

WK

WV

WO
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

print("\nQ Shape")

print(Q.shape)

print("\nK Shape")

print(K.shape)

print("\nV Shape")

print(V.shape)

# ============================================================
# Step 5 : Split Into Multiple Heads
# ============================================================

"""
Current

(3,512)

↓

(3,8,64)
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
"""

Q = np.transpose(Q, (1, 0, 2))

K = np.transpose(K, (1, 0, 2))

V = np.transpose(V, (1, 0, 2))

print("\nAfter Transpose")

print(Q.shape)

# ============================================================
# Step 7 : Softmax
# ============================================================

def softmax(x):

    x = x - np.max(x, axis=-1, keepdims=True)

    exp = np.exp(x)

    return exp / np.sum(exp, axis=-1, keepdims=True)

# ============================================================
# Step 8 : Multi-Head Attention
# ============================================================

head_outputs = []

for head in range(num_heads):

    q = Q[head]

    k = K[head]

    v = V[head]

    # ----------------------------------------
    # Compute Attention Scores
    # ----------------------------------------

    scores = q @ k.T

    # ----------------------------------------
    # Scale Scores
    # ----------------------------------------

    scores = scores / math.sqrt(dk)

    # ----------------------------------------
    # Softmax
    # ----------------------------------------

    attention_weights = softmax(scores)

    # ----------------------------------------
    # Weighted Sum
    # ----------------------------------------

    output = attention_weights @ v

    head_outputs.append(output)

print("\nNumber of Heads")

print(len(head_outputs))

# ============================================================
# Step 9 : Concatenate Heads
# ============================================================

"""
Current

(8,3,64)

↓

(3,512)
"""

head_outputs = np.stack(head_outputs)

head_outputs = np.transpose(head_outputs, (1,0,2))

multi_head_output = head_outputs.reshape(sequence_length, d_model)

print("\nConcatenated Shape")

print(multi_head_output.shape)

# ============================================================
# Step 10 : Final Linear Projection
# ============================================================

"""
Concat

↓

WO

↓

Output
"""

multi_head_output = multi_head_output @ WO

print("\nMulti-Head Output Shape")

print(multi_head_output.shape)

# ============================================================
# Step 11 : First Residual Connection
# ============================================================

"""
Residual

Input

+

Multi-Head Output
"""

residual1 = X + multi_head_output

print("\nAfter Residual Connection")

print(residual1.shape)

# ============================================================
# Step 12 : Layer Normalization
# ============================================================

"""
Compute Mean

for every word
"""

mean1 = np.mean(residual1,
                axis=1,
                keepdims=True)

"""
Compute Variance
"""

variance1 = np.var(residual1,
                   axis=1,
                   keepdims=True)

"""
Normalize
"""

normalized1 = (residual1 - mean1) / np.sqrt(variance1 + epsilon)

"""
Learnable Parameters

Gamma

Beta
"""

gamma1 = np.ones(d_model)

beta1 = np.zeros(d_model)

encoder_input = gamma1 * normalized1 + beta1

print("\nAfter First LayerNorm")

print(encoder_input.shape)

# ============================================================
# End of Part 1
# ============================================================

print("\nFirst Half of Encoder Layer Completed Successfully!")

"""
The next part implements

1. Feed Forward Network

2. Second Residual Connection

3. Second Layer Normalization

4. Final Encoder Output
"""

# ============================================================
# Step 13 : Feed Forward Network Parameters
# ============================================================

"""
The Feed Forward Network expands

512

↓

2048

↓

512

Every token is processed independently.
"""

W1 = np.random.randn(d_model, d_ff)

b1 = np.random.randn(d_ff)

W2 = np.random.randn(d_ff, d_model)

b2 = np.random.randn(d_model)


# ============================================================
# Step 14 : First Linear Layer
# ============================================================

"""
Equation

XW1 + b1

Shape

(3,512)

×

(512,2048)

↓

(3,2048)
"""

hidden = encoder_input @ W1 + b1

print("\nAfter First Linear Layer")

print(hidden.shape)


# ============================================================
# Step 15 : ReLU Activation
# ============================================================

"""
ReLU

Negative values

↓

0

Positive values

↓

Remain unchanged
"""

hidden = np.maximum(0, hidden)

print("\nAfter ReLU")

print(hidden.shape)


# ============================================================
# Step 16 : Second Linear Layer
# ============================================================

"""
Equation

hiddenW2 + b2

Shape

(3,2048)

×

(2048,512)

↓

(3,512)
"""

ffn_output = hidden @ W2 + b2

print("\nFeed Forward Output Shape")

print(ffn_output.shape)


# ============================================================
# Step 17 : Second Residual Connection
# ============================================================

"""
Residual

Encoder Input

+

Feed Forward Output
"""

residual2 = encoder_input + ffn_output

print("\nAfter Second Residual Connection")

print(residual2.shape)


# ============================================================
# Step 18 : Second Layer Normalization
# ============================================================

"""
Compute Mean

for every word
"""

mean2 = np.mean(
    residual2,
    axis=1,
    keepdims=True
)

"""
Compute Variance
"""

variance2 = np.var(
    residual2,
    axis=1,
    keepdims=True
)

"""
Normalize
"""

normalized2 = (
    residual2 - mean2
) / np.sqrt(
    variance2 + epsilon
)

"""
Learnable Parameters

Gamma

Beta
"""

gamma2 = np.ones(d_model)

beta2 = np.zeros(d_model)

encoder_output = gamma2 * normalized2 + beta2

print("\nFinal Encoder Output Shape")

print(encoder_output.shape)


# ============================================================
# Step 19 : Display Results
# ============================================================

print("\nFirst 10 values of Word 1")

print(encoder_output[0][:10])

print("\nFirst 10 values of Word 2")

print(encoder_output[1][:10])

print("\nFirst 10 values of Word 3")

print(encoder_output[2][:10])


# ============================================================
# Step 20 : Summary
# ============================================================

"""
Complete Encoder Layer

Input

(3,512)

↓

Multi-Head Attention

↓

Linear Projection

↓

Residual Connection

↓

Layer Normalization

↓

Feed Forward Network

512

↓

2048

↓

512

↓

Residual Connection

↓

Layer Normalization

↓

Final Encoder Output

(3,512)
"""

print("\nEncoder Layer Completed Successfully!")