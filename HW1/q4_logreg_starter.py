"""
Q4 starter: Logistic Regression with Feature Engineering

Submit this file as q4_logreg_starter.py.

You may use NumPy only for the model implementation. Do not use
sklearn or automatic differentiation.

Required public interface:
    featurize(X_raw)
    LogisticRegression(...).fit(X, y)
    LogisticRegression(...).predict_proba(X)
    LogisticRegression(...).predict(X)
    LogisticRegression(...)._loss_and_grad(X, y, w, b)

_loss_and_grad evaluates the regularised loss and its gradients at an
arbitrary (w, b), independent of self.w / self.b. fit() should call this
in its gradient-descent loop; the public tests also call it directly with
finite differences to check your gradients, the same way Q2 does.
"""

import numpy as np


def sigmoid(z):
    """Numerically stable logistic sigmoid."""
    # TODO: implement
    pass


def featurize(X_raw):
    """
    Convert the raw columns

        [sigma_xx, sigma_yy, sigma_zz,
         tau_xy, tau_yz, tau_zx, sigma_Y]

    into the 10 engineered features

        [sigma_xx^2, sigma_yy^2, sigma_zz^2,
         tau_xy^2, tau_yz^2, tau_zx^2,
         sigma_xx*sigma_yy, sigma_yy*sigma_zz,
         sigma_zz*sigma_xx, sigma_Y^2]

    X_raw may be an (N, 7) NumPy array or an equivalent array-like object.
    Return an (N, 10) NumPy array.

    Do not use temperature_c, strain_rate, or batch_id here.
    """
    # TODO: implement
    pass


class LogisticRegression:
    """
    Binary logistic regression trained by gradient descent.

    The model is

        p = sigmoid(X @ w + b)

    with binary cross-entropy and L2 regularisation on w only.

    Parameters
    ----------
    lr : float
        Learning rate.
    n_iter : int
        Number of gradient-descent iterations.
    lam : float
        L2 regularisation strength.
    seed : int
        Random seed for parameter initialisation.
    """

    def __init__(self, lr=1e-4, n_iter=5000, lam=0.0, seed=0):
        self.lr = lr
        self.n_iter = n_iter
        self.lam = lam
        self.seed = seed

        self.w = None
        self.b = None
        self.loss_history = None

    def _loss_and_grad(self, X, y, w, b):
        """
        Evaluate the mean regularised binary cross-entropy loss and its
        gradients at the given (w, b) -- NOT self.w / self.b.

        loss = mean BCE(sigmoid(X @ w + b), y) + (lam / 2) * ||w||^2

        L2 regularisation applies to w only, never to b.

        Parameters
        ----------
        X : array-like, shape (N, d)
        y : array-like, shape (N,)
        w : array-like, shape (d,)
        b : float

        Returns
        -------
        loss : float
        grad_w : ndarray, shape (d,)
        grad_b : float
        """
        # TODO: implement.
        # This is exactly the logistic-regression special case of the
        # dL/ds = p - y result from Q2: grad_w = X.T @ (p - y) / N + lam*w,
        # grad_b = mean(p - y).
        pass

    def fit(self, X, y):
        """
        Fit the model using full-batch gradient descent.

        Use _loss_and_grad(X, y, self.w, self.b) inside the update loop
        so that your gradient descent and your gradient checker are
        guaranteed to use the same formula.

        Parameters
        ----------
        X : array-like, shape (N, d)
        y : array-like, shape (N,)
            Binary labels in {0, 1}.

        Returns
        -------
        self
        """
        # TODO: implement
        pass

    def predict_proba(self, X):
        """
        Return P(y=1 | X), shape (N,).
        """
        # TODO: implement
        pass

    def predict(self, X):
        """
        Return binary predictions using a threshold of 0.5.
        """
        # TODO: implement
        pass