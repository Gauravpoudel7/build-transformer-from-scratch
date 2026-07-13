"""
============================================================
Transformer From Scratch - Part 7
Encoder Layer
============================================================

Paper:
Attention Is All You Need
Section 3.1

One complete Transformer Encoder Layer.

Pipeline

Input
    │
    ▼
Multi-Head Attention
    │
    ▼
Add & LayerNorm
    │
    ▼
Feed Forward Network
    │
    ▼
Add & LayerNorm
    │
    ▼
Encoder Output
"""

import numpy as np
import math


class EncoderLayer:

    def __init__(self,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 epsilon=1e-5):

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.dv = self.dk
        self.d_ff = d_ff
        self.epsilon = epsilon

        # -------------------------
        # Multi-Head Attention Weights
        # -------------------------

        self.WQ = np.random.randn(d_model, d_model)
        self.WK = np.random.randn(d_model, d_model)
        self.WV = np.random.randn(d_model, d_model)
        self.WO = np.random.randn(d_model, d_model)

        # -------------------------
        # Feed Forward Network
        # -------------------------

        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.random.randn(d_ff)

        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.random.randn(d_model)

        # -------------------------
        # LayerNorm Parameters
        # -------------------------

        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)

        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    # =========================================================
    # Softmax
    # =========================================================

    def softmax(self, x):

        x = x - np.max(x, axis=-1, keepdims=True)

        exp = np.exp(x)

        return exp / np.sum(exp, axis=-1, keepdims=True)

    # =========================================================
    # Layer Normalization
    # =========================================================

    def layer_norm(self, x, gamma, beta):

        mean = np.mean(x, axis=1, keepdims=True)

        variance = np.var(x, axis=1, keepdims=True)

        normalized = (x - mean) / np.sqrt(
            variance + self.epsilon
        )

        return gamma * normalized + beta

    # =========================================================
    # Multi-Head Attention
    # =========================================================

    def multi_head_attention(self, X):

        Q = X @ self.WQ
        K = X @ self.WK
        V = X @ self.WV

        seq_len = X.shape[0]

        Q = Q.reshape(seq_len,
                      self.num_heads,
                      self.dk)

        K = K.reshape(seq_len,
                      self.num_heads,
                      self.dk)

        V = V.reshape(seq_len,
                      self.num_heads,
                      self.dv)

        Q = np.transpose(Q, (1, 0, 2))
        K = np.transpose(K, (1, 0, 2))
        V = np.transpose(V, (1, 0, 2))

        head_outputs = []

        for head in range(self.num_heads):

            q = Q[head]
            k = K[head]
            v = V[head]

            scores = q @ k.T

            scores = scores / math.sqrt(self.dk)

            weights = self.softmax(scores)

            output = weights @ v

            head_outputs.append(output)

        head_outputs = np.stack(head_outputs)

        head_outputs = np.transpose(
            head_outputs,
            (1, 0, 2)
        )

        concat = head_outputs.reshape(
            seq_len,
            self.d_model
        )

        return concat @ self.WO

    # =========================================================
    # Feed Forward Network
    # =========================================================

    def feed_forward(self, X):

        hidden = X @ self.W1 + self.b1

        hidden = np.maximum(0, hidden)

        output = hidden @ self.W2 + self.b2

        return output

    # =========================================================
    # Forward Pass
    # =========================================================

    def forward(self, X):

        # -------------------------
        # Multi-Head Attention
        # -------------------------

        attention = self.multi_head_attention(X)

        # -------------------------
        # Add & LayerNorm
        # -------------------------

        X = self.layer_norm(
            X + attention,
            self.gamma1,
            self.beta1
        )

        # -------------------------
        # Feed Forward
        # -------------------------

        ffn = self.feed_forward(X)

        # -------------------------
        # Add & LayerNorm
        # -------------------------

        X = self.layer_norm(
            X + ffn,
            self.gamma2,
            self.beta2
        )

        return X


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)

    sequence_length = 3

    d_model = 512

    X = np.random.randn(
        sequence_length,
        d_model
    )

    encoder = EncoderLayer()

    output = encoder.forward(X)

    print("Input Shape")

    print(X.shape)

    print("\nOutput Shape")

    print(output.shape)

    print("\nFirst 10 values of Word 1")

    print(output[0][:10])

    print("\nEncoder Layer Completed Successfully!")