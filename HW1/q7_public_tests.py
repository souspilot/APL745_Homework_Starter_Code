"""
APL 745 -- Question 7 public tests

The tests check:
    1. forward pass
    2. loss
    3. analytical gradients using finite differences
    4. batch-gradient consistency
"""

import numpy as np

from HW1.q7_backprop_starter import NeuralNetwork, softplus, sigmoid


def numerical_gradient(model, X, y, name, h=1e-6):
    param = model.params[name]
    grad = np.zeros_like(param, dtype=float)

    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = param[idx]

        param[idx] = old + h
        loss_plus = model.loss(X, y)

        param[idx] = old - h
        loss_minus = model.loss(X, y)

        param[idx] = old
        grad[idx] = (loss_plus - loss_minus) / (2 * h)

        it.iternext()

    return grad


def relative_error(a, b):
    return (
        np.linalg.norm(a - b)
        / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)
    )


def set_worked_example(model):
    model.params["w1"] = np.array([0.4, -0.2, 0.3])
    model.params["b1"] = np.array(0.1)
    model.params["v"] = np.array(0.2)
    model.params["w_skip"] = np.array([0.15, 0.05, -0.1])
    model.params["b2"] = np.array(-0.05)
    model.params["w_out"] = np.array(0.6)
    model.params["b_out"] = np.array(0.05)


def test_forward():
    model = NeuralNetwork(d=3)
    set_worked_example(model)

    X = np.array([[0.5, -0.3, 0.8]])
    p = model.forward(X)

    expected = 0.6759915836
    assert np.allclose(p, expected, atol=1e-8)

    print("Forward pass: PASS")


def test_loss():
    model = NeuralNetwork(d=3)
    set_worked_example(model)

    X = np.array([[0.5, -0.3, 0.8]])
    y = np.array([1.0])

    expected = 0.3915746533
    assert np.allclose(model.loss(X, y), expected, atol=1e-8)

    print("Loss: PASS")


def test_gradients():
    rng = np.random.default_rng(1)

    model = NeuralNetwork(d=3, seed=2)
    X = rng.normal(size=(4, 3))
    y = np.array([0.0, 1.0, 1.0, 0.0])

    analytical = model.backward(X, y)

    for name in model.params:
        numerical = numerical_gradient(model, X, y, name)
        error = relative_error(analytical[name], numerical)

        print(f"{name:8s}: relative error = {error:.3e}")
        assert error < 1e-5

    print("Gradient check: PASS")


def test_batch_gradients():
    rng = np.random.default_rng(3)

    model = NeuralNetwork(d=3, seed=4)
    X = rng.normal(size=(5, 3))
    y = rng.integers(0, 2, size=5).astype(float)

    batch_grads = model.backward(X, y)

    for name in model.params:
        individual = []

        for i in range(len(X)):
            g = model.backward(X[i:i+1], y[i:i+1])[name]
            individual.append(g)

        mean_grad = np.mean(individual, axis=0)
        assert np.allclose(batch_grads[name], mean_grad, atol=1e-10)

    print("Batch gradients: PASS")


if __name__ == "__main__":
    test_forward()
    test_loss()
    test_gradients()
    test_batch_gradients()
    print("\nAll public tests passed.")