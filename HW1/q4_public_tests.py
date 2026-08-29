"""
Public tests for Q4.

Run:
    python q4_public_tests.py

The tests import the student's q4_logreg_starter.py and check:
1. the required feature map;
2. basic model API and probability output;
3. that logistic regression can learn the engineered von Mises boundary
   on a deterministic validation dataset.

The validation dataset is generated deterministically here so that no
third data file is required.
"""

import numpy as np

from HW1.q4_logreg_starter import LogisticRegression, featurize


def make_validation_data(n=600, seed=123):
    """
    Deterministic synthetic validation dataset.

    Columns:
        sigma_xx, sigma_yy, sigma_zz,
        tau_xy, tau_yz, tau_zx, sigma_Y

    Labels are generated from the von Mises criterion with a small amount
    of noise near the boundary.
    """
    rng = np.random.default_rng(seed)

    stress = rng.normal(0.0, 1.0, size=(n, 6))
    sigma_y = rng.uniform(0.8, 1.8, size=n)

    sx, sy, sz, txy, tyz, tzx = stress.T

    vm2 = (
        sx**2 + sy**2 + sz**2
        - sx * sy - sy * sz - sz * sx
        + 3.0 * (txy**2 + tyz**2 + tzx**2)
    )

    # Add a small deterministic label-noise region around the boundary.
    y = (vm2 >= sigma_y**2).astype(int)
    near_boundary = np.abs(vm2 - sigma_y**2) < 0.08
    noise = rng.random(n) < 0.10
    y[near_boundary & noise] ^= 1

    X_raw = np.column_stack([stress, sigma_y])
    return X_raw, y


def accuracy(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def test_feature_map():
    X_raw = np.array([
        [2.0, 3.0, 4.0, 0.5, -1.0, 2.0, 5.0],
        [-1.0, 2.0, 0.5, 2.0, 3.0, -0.5, 1.5],
    ])

    X = featurize(X_raw)

    expected = np.array([
        [4.0, 9.0, 16.0, 0.25, 1.0, 4.0,
         6.0, 12.0, 8.0, 25.0],
        [1.0, 4.0, 0.25, 4.0, 9.0, 0.25,
         -2.0, 1.0, -0.5, 2.25],
    ])

    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 10)
    assert np.allclose(X, expected)

    print("PASS: feature map")


def test_model_api():
    X_raw, y = make_validation_data(n=80)
    X = featurize(X_raw)

    model = LogisticRegression(lr=1e-3, n_iter=1000, lam=1e-4, seed=0)
    result = model.fit(X, y)

    assert result is model
    assert model.w is not None
    assert model.b is not None
    assert model.w.shape == (10,)

    p = model.predict_proba(X)
    pred = model.predict(X)

    assert p.shape == (80,)
    assert pred.shape == (80,)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all((pred == 0) | (pred == 1))

    print("PASS: model API")


def test_engineered_model_accuracy():
    """
    The engineered representation should make the von Mises boundary
    straightforward for logistic regression to learn.
    """
    X_raw, y = make_validation_data(n=600)

    # Fixed split: students are not being tested on split selection here.
    X_train_raw, X_val_raw = X_raw[:450], X_raw[450:]
    y_train, y_val = y[:450], y[450:]

    X_train = featurize(X_train_raw)
    X_val = featurize(X_val_raw)

    model = LogisticRegression(
        lr=1e-3,
        n_iter=5000,
        lam=1e-4,
        seed=0,
    )
    model.fit(X_train, y_train)

    val_acc = accuracy(y_val, model.predict(X_val))

    # This threshold leaves room for optimisation/finite-sample variation
    # while still detecting a non-functional implementation.
    assert val_acc >= 0.80, (
        f"Validation accuracy too low: {val_acc:.3f}. "
        "Check featurize(), sigmoid(), gradients, and gradient descent."
    )

    print(f"PASS: engineered-model validation accuracy = {val_acc:.3f}")


def main():
    tests = [
        test_feature_map,
        test_model_api,
        test_engineered_model_accuracy,
    ]

    for test in tests:
        test()

    print("\nAll public tests passed.")


if __name__ == "__main__":
    main()
