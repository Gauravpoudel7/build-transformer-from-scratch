"""
============================================================
Transformer From Scratch - Part 6
Residual Connection + Layer Normalization
============================================================

Paper:
Attention Is All You Need
Section 3.1

Every sub-layer inside the Transformer Encoder
and Decoder is wrapped with

Residual Connection

+

Layer Normalization

Equation

Output = LayerNorm(Input + Sublayer(Input))

Pipeline

Input
   │
   ▼
Sub-layer Output
   │
   ▼
Residual Addition
(Input + Output)
   │
   ▼
Layer Normalization
   │
   ▼
Final Output
"""

import numpy as np

# ============================================================
# Step 1 : Model Parameters
# ============================================================

"""
Example sentence

"I love Transformers"

Sequence Length = 3

Each word has

512 features.
"""

sequence_length = 3

d_model = 512

epsilon = 1e-5

np.random.seed(42)


# ============================================================
# Step 2 : Create Example Input
# ============================================================

"""
Normally this comes from

Embedding + Positional Encoding

Shape

(3,512)
"""

X = np.random.randn(sequence_length, d_model)

print("Input Shape")

print(X.shape)


# ============================================================
# Step 3 : Example Sub-layer Output
# ============================================================

"""
Imagine this is the output from

Multi-Head Attention

or

Feed Forward Network

Shape

(3,512)
"""

sublayer_output = np.random.randn(sequence_length, d_model)

print("\nSub-layer Output Shape")

print(sublayer_output.shape)


# ============================================================
# Step 4 : Residual Connection
# ============================================================

"""
Residual Connection

Output

=

Input

+

Sub-layer Output

The shapes must be identical.

(3,512)

+

(3,512)

↓

(3,512)
"""

residual = X + sublayer_output

print("\nAfter Residual Connection")

print(residual.shape)


# ============================================================
# Step 5 : Compute Mean
# ============================================================

"""
Layer Normalization normalizes

each token independently.

For every word,

compute the mean across

its 512 features.

Result

(3,1)
"""

mean = np.mean(residual, axis=1, keepdims=True)

print("\nMean Shape")

print(mean.shape)


# ============================================================
# Step 6 : Compute Variance
# ============================================================

"""
Variance measures

how spread out the values are.

Again,

computed separately

for each word.
"""

variance = np.var(residual, axis=1, keepdims=True)

print("\nVariance Shape")

print(variance.shape)


# ============================================================
# Step 7 : Normalize
# ============================================================

"""
Equation

(x - mean)

---------------

sqrt(variance + epsilon)

epsilon prevents

division by zero.
"""

normalized = (residual - mean) / np.sqrt(variance + epsilon)

print("\nNormalized Shape")

print(normalized.shape)


# ============================================================
# Step 8 : Learnable Scale and Shift
# ============================================================

"""
LayerNorm has

two learnable parameters

Gamma (γ)

controls scaling

Beta (β)

controls shifting
"""

gamma = np.ones(d_model)

beta = np.zeros(d_model)

output = gamma * normalized + beta

print("\nFinal Output Shape")

print(output.shape)


# ============================================================
# Step 9 : Display Results
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

Sub-layer

(3,512)

↓

Residual Addition

(3,512)

↓

Layer Normalization

↓

Final Output

(3,512)
"""

print("\nResidual Connection + LayerNorm Completed Successfully!")