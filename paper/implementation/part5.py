"""
============================================================
Transformer From Scratch - Part 5
Position-wise Feed Forward Network (FFN)
============================================================

Paper:
Attention Is All You Need
Section 3.3

The Feed Forward Network is applied to every token
independently after Multi-Head Attention.

Formula

FFN(x) = max(0, xW1 + b1)W2 + b2

Pipeline

Multi-Head Attention Output
            │
            ▼
Linear Layer (512 → 2048)
            │
            ▼
ReLU
            │
            ▼
Linear Layer (2048 → 512)
            │
            ▼
FFN Output
"""

import numpy as np

# ============================================================
# Step 1 : Model Parameters
# ============================================================

"""
Example sentence

"I love Transformers"

Sequence Length = 3

Each word has 512 features.

The hidden layer expands the features
from 512 to 2048.
"""

sequence_length = 3

d_model = 512

d_ff = 2048

np.random.seed(42)


# ============================================================
# Step 2 : Example Input
# ============================================================

"""
Normally this input comes from

Multi-Head Attention

Shape

(3,512)
"""

X = np.random.randn(sequence_length, d_model)

print("Input Shape")

print(X.shape)


# ============================================================
# Step 3 : Create Learnable Parameters
# ============================================================

"""
First Linear Layer

512

↓

2048
"""

W1 = np.random.randn(d_model, d_ff)

b1 = np.random.randn(d_ff)


"""
Second Linear Layer

2048

↓

512
"""

W2 = np.random.randn(d_ff, d_model)

b2 = np.random.randn(d_model)


# ============================================================
# Step 4 : First Linear Layer
# ============================================================

"""
Equation

XW1 + b1

Shapes

(3,512)

×

(512,2048)

↓

(3,2048)
"""

hidden = X @ W1 + b1

print("\nAfter First Linear Layer")

print(hidden.shape)


# ============================================================
# Step 5 : ReLU Activation
# ============================================================

"""
ReLU

Negative values become zero.

Positive values remain unchanged.

Example

-3 → 0

5 → 5
"""

hidden = np.maximum(0, hidden)

print("\nAfter ReLU")

print(hidden.shape)


# ============================================================
# Step 6 : Second Linear Layer
# ============================================================

"""
Equation

hiddenW2 + b2

Shapes

(3,2048)

×

(2048,512)

↓

(3,512)
"""

output = hidden @ W2 + b2

print("\nOutput Shape")

print(output.shape)


# ============================================================
# Step 7 : Display Results
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

Linear

512 → 2048

↓

ReLU

↓

Linear

2048 → 512

↓

Output

(3,512)
"""

print("\nFeed Forward Network Completed Successfully!")