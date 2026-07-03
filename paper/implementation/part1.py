"""
============================================================
Transformer From Scratch - Part 1
Token Embedding Layer
============================================================

This file implements the first component of the Transformer:
    1. Token Embedding

The purpose of the embedding layer is to convert each token
(integer ID) into a dense vector representation.

Paper:
"Attention Is All You Need"
Section 3.4 - Embeddings and Softmax
"""

import numpy as np
import math


# ============================================================
# Step 1: Define the vocabulary
# ============================================================

"""
Every word (or subword token) is assigned a unique integer ID.

Special tokens:
<PAD> : Used for padding shorter sentences
<SOS> : Start Of Sentence
<EOS> : End Of Sentence
"""

vocabulary = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "I": 3,
    "love": 4,
    "Transformers": 5,
    "NLP": 6
}

# Total number of words/tokens
vocab_size = len(vocabulary)

print("Vocabulary Size:", vocab_size)


# ============================================================
# Step 2: Model Dimension
# ============================================================

"""
The Transformer paper uses

d_model = 512

This means every word is represented using
512 floating point numbers.
"""

d_model = 512


# ============================================================
# Step 3: Create the Embedding Matrix
# ============================================================

"""
The embedding matrix has shape

(vocab_size, d_model)

For our example:

(7, 512)

Each row corresponds to one token.

Initially these vectors are random.

During training these values are updated by
backpropagation.
"""

embedding_matrix = np.random.randn(vocab_size, d_model)

print("\nEmbedding Matrix Shape:")
print(embedding_matrix.shape)


# ============================================================
# Step 4: Convert a Sentence into Token IDs
# ============================================================

"""
Sentence:

I love Transformers

Tokenizer converts it into

[3,4,5]
"""

sentence = ["I", "love", "Transformers"]

token_ids = [vocabulary[word] for word in sentence]

print("\nSentence:")
print(sentence)

print("\nToken IDs:")
print(token_ids)


# ============================================================
# Step 5: Embedding Lookup
# ============================================================

"""
Instead of processing the words directly,
we retrieve their vectors.

This is simply selecting rows from the
embedding matrix.

embedding_matrix[token_ids]

Shape becomes

(number_of_tokens, d_model)

(3,512)
"""

embedded_tokens = embedding_matrix[token_ids]

print("\nEmbedded Token Shape:")
print(embedded_tokens.shape)


# ============================================================
# Step 6: Scale the Embeddings
# ============================================================

"""
The Transformer paper multiplies the embeddings by

sqrt(d_model)

For d_model = 512

sqrt(512) ≈ 22.627

Why?

Without scaling, the embeddings have relatively
small values.

Scaling keeps the embedding magnitude compatible
with positional encoding.
"""

scale = math.sqrt(d_model)

embedded_tokens = embedded_tokens * scale

print("\nScaling Factor:")
print(scale)


# ============================================================
# Step 7: Inspect One Embedding
# ============================================================

"""
Each word now has a vector of length 512.

Let's inspect the first few values
for the word "I".
"""

print("\nEmbedding Vector for 'I'")
print(embedded_tokens[0][:10])  # first 10 numbers


# ============================================================
# Step 8: Final Output
# ============================================================

"""
Final tensor shape

(number_of_words, d_model)

This is exactly what will be sent to the
Positional Encoding layer.
"""

print("\nFinal Output Shape:")
print(embedded_tokens.shape)