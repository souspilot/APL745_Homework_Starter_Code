"""
APL 745 -- Question 3 starter code
Backpropagation on a Neural Network with a Convex Hidden Path

Implement:
    forward()
    loss()
    backward()

Use the equations in the question. The hidden path is the scalar
architecture used in Parts A-D; inputs are supplied as batches.
"""

import numpy as np


def softplus(x):
    """Numerically stable softplus."""
    return np.logaddexp(0.0, x)


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class NeuralNetwork:
    def __init__(self, d, seed=0):
        rng = np.random.default_rng(seed)

        self.d = d

        self.params = {
            "w1": rng.normal(0.0, 0.1, size=d),
            "b1": np.array(0.0),
            "v": np.array(rng.normal(0.0, 0.1)),
            "w_skip": rng.normal(0.0, 0.1, size=d),
            "b2": np.array(0.0),
            "w_out": np.array(rng.normal(0.0, 0.1)),
            "b_out": np.array(0.0),
        }

    def forward(self, X):
        """
        X has shape (N, d).

        Return:
            p: predicted probabilities, shape (N,)
        """
        # TODO: implement the forward pass.
        pass

    def loss(self, X, y):
        """Return the mean binary cross-entropy loss."""
        # TODO: implement the loss.
        pass

    def backward(self, X, y):
        """
        Return the gradients of the mean loss with respect to
        all parameters in self.params.

        Derive these gradients in Part B before implementing them.
        """
        # TODO: implement the backward pass.
        pass
