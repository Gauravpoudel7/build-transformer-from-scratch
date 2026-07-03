"""
============================================================
Transformer From Scratch - Part 2
Positional Encoding
============================================================

Paper:
Attention Is All You Need
Section 3.5

Purpose:
--------
Since the Transformer has no RNN or CNN, it has no idea
about the order of words.

To solve this, we generate a positional encoding matrix
using sine and cosine functions and add it to the token
embeddings.

Final Input = Token Embedding + Positional Encoding
"""

import numpy as np
import math

# ============================================================
# Step 1 : Define Model Parameters
# ============================================================

"""
Maximum sequence length.

This is the maximum number of words the model
can process at one time.

Example:
If max_sequence_length = 10,
we can encode sentences containing up to
10 tokens.
"""

max_sequence_length = 10

"""
Embedding dimension from the paper.

Every word is represented using
512 numbers.
"""

d_model = 512


# ============================================================
# Step 2 : Create Empty Positional Encoding Matrix
# ============================================================

"""
Shape:

(max_sequence_length, d_model)

For us:

(10,512)

Each row corresponds to one position
in the sentence.

Position 0
Position 1
Position 2
...
Position 9
"""

PE = np.zeros((max_sequence_length, d_model))


# ============================================================
# Step 3 : Compute Positional Encoding
# ============================================================

"""
Paper formulas

For EVEN dimensions

PE(pos,2i) =
sin(pos / 10000^(2i/d_model))

For ODD dimensions

PE(pos,2i+1) =
cos(pos / 10000^(2i/d_model))
"""

for pos in range(max_sequence_length):

    # Loop over embedding dimensions
    for i in range(0, d_model, 2):

        # Compute denominator
        denominator = math.pow(10000, i / d_model)

        # Even index -> sin
        PE[pos, i] = math.sin(pos / denominator)

        # Odd index -> cos
        if i + 1 < d_model:
            PE[pos, i + 1] = math.cos(pos / denominator)


# ============================================================
# Step 4 : Display Matrix Information
# ============================================================

print("Positional Encoding Shape:")
print(PE.shape)

print("\nFirst 10 values of Position 0")
print(PE[0][:10])

print("\nFirst 10 values of Position 1")
print(PE[1][:10])

print("\nFirst 10 values of Position 2")
print(PE[2][:10])


# ============================================================
# Step 5 : Example Embedding Matrix
# ============================================================

"""
Suppose the sentence is

"I love Transformers"

There are

3 words

Each word has 512 values.

Shape

(3,512)

Normally this comes from Part 1
(Token Embedding).

Here we create random embeddings
just for demonstration.
"""

embedded_tokens = np.random.randn(3, d_model)

print("\nEmbedding Shape:")
print(embedded_tokens.shape)


# ============================================================
# Step 6 : Extract Required Positional Encodings
# ============================================================

"""
Our sentence has only

3 words

So we only need

Position 0
Position 1
Position 2

Shape

(3,512)
"""

sentence_length = embedded_tokens.shape[0]

position_vectors = PE[:sentence_length]

print("\nPosition Vector Shape:")
print(position_vectors.shape)


# ============================================================
# Step 7 : Add Positional Encoding
# ============================================================

"""
Paper:

Input = Embedding + Positional Encoding

Both matrices have shape

(3,512)

So element-wise addition is possible.
"""

transformer_input = embedded_tokens + position_vectors


# ============================================================
# Step 8 : Final Result
# ============================================================

print("\nTransformer Input Shape:")
print(transformer_input.shape)

print("\nFirst 10 values of First Word")
print(transformer_input[0][:10])

print("\nFirst 10 values of Second Word")
print(transformer_input[1][:10])

print("\nFirst 10 values of Third Word")
print(transformer_input[2][:10])


# ============================================================
# Step 9 : Visualization
# ============================================================

"""
Sentence

"I love Transformers"

↓

Embedding Matrix

Shape

(3,512)

↓

Positional Encoding

Position 0
Position 1
Position 2

↓

Element-wise Addition

↓

Final Input

(3,512)

↓

Send to Encoder
"""