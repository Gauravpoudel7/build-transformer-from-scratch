"""
============================================================
Transformer From Scratch - Part 7
Encoder Stack
============================================================

Paper:
Attention Is All You Need
Section 3.1

The original Transformer does not use just one
Encoder Layer.

Instead, it stacks N identical Encoder Layers.

Original Paper

N = 6

Pipeline

Input
   │
   ▼
Encoder Layer 1
   │
   ▼
Encoder Layer 2
   │
   ▼
Encoder Layer 3
   │
   ▼
Encoder Layer 4
   │
   ▼
Encoder Layer 5
   │
   ▼
Encoder Layer 6
   │
   ▼
Final Encoder Output
"""

import numpy as np

# Import the Encoder Layer from Part 7
from refactoredPart6 import EncoderLayer


# ============================================================
# Encoder Stack
# ============================================================

class EncoderStack:
    """
    A Transformer Encoder consists of multiple
    identical Encoder Layers.

    Each layer has its own trainable parameters.

    The output of one layer becomes
    the input of the next layer.
    """

    def __init__(
        self,
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
    ):

        self.num_layers = num_layers

        # ----------------------------------------------------
        # Create N independent Encoder Layers
        # ----------------------------------------------------

        self.layers = []

        for i in range(num_layers):

            layer = EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )

            self.layers.append(layer)

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(self, X):

        print("\n==============================")
        print("Starting Encoder Stack")
        print("==============================")

        # ----------------------------------------------------
        # Pass the input through every encoder layer
        # ----------------------------------------------------

        for index, layer in enumerate(self.layers):

            print(f"\nPassing through Encoder Layer {index + 1}")

            X = layer.forward(X)

            print("Output Shape :", X.shape)

        print("\n==============================")
        print("Encoder Stack Finished")
        print("==============================")

        return X


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)

    # --------------------------------------------------------
    # Example Sentence
    #
    # "I love Transformers"
    #
    # 3 words
    # 512-dimensional embedding
    # --------------------------------------------------------

    sequence_length = 3

    d_model = 512

    X = np.random.randn(sequence_length, d_model)

    print("Input Shape")

    print(X.shape)

    # --------------------------------------------------------
    # Build Encoder Stack
    # --------------------------------------------------------

    encoder = EncoderStack(
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
    )

    # --------------------------------------------------------
    # Run Forward Pass
    # --------------------------------------------------------

    output = encoder.forward(X)

    # --------------------------------------------------------
    # Display Results
    # --------------------------------------------------------

    print("\n=================================")
    print("Final Encoder Output Shape")
    print("=================================")

    print(output.shape)

    print("\nFirst 10 values of Word 1")

    print(output[0][:10])

    print("\nFirst 10 values of Word 2")

    print(output[1][:10])

    print("\nFirst 10 values of Word 3")

    print(output[2][:10])

    print("\nEncoder Stack Completed Successfully!")


# ============================================================
# Summary
# ============================================================

"""
Transformer Encoder

Input

(3,512)

        │

        ▼

Encoder Layer 1

        │

        ▼

Encoder Layer 2

        │

        ▼

Encoder Layer 3

        │

        ▼

Encoder Layer 4

        │

        ▼

Encoder Layer 5

        │

        ▼

Encoder Layer 6

        │

        ▼

Final Encoder Output

(3,512)

Every encoder layer has

• Multi-Head Attention

↓

• Add & LayerNorm

↓

• Feed Forward Network

↓

• Add & LayerNorm

The output of one encoder layer
becomes the input to the next.

The original Transformer paper
uses N = 6 encoder layers.
"""