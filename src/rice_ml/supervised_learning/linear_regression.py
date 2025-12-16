"""
Linear regression implementation for the rice_ml package.

This module provides a simple, user-friendly API for ordinary least squares
linear regression. It is implemented from scratch using NumPy only
(no scikit-learn dependency) so that students can read and understand
the core ideas.

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.linear_regression import LinearRegression
>>>
>>> X = np.array([[1, 2],
...               [2, 3],
...               [4, 5]])
>>> y = np.array([2, 3, 5])
>>>
>>> model = LinearRegression()
>>> model.fit(X, y)
>>> model.predict(X)
array([2., 3., 5.])
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class LinearRegression:
    """Simple linear regression using ordinary least squares.

    This implementation mirrors high-level APIs but is intentionally
    compact for teaching purposes.

    Attributes
    ----------
    coef_ : np.ndarray
        Learned feature coefficients of shape (n_features,).
    intercept_ : float
        Learned intercept term.
    """

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the linear regression model on the given training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 2:
            raise ValueError("y must be a 1D array of shape (n_samples,) or 2D column vector.")

        n_samples, n_features = X.shape

        # Add bias term
        X_bias = np.hstack([np.ones((n_samples, 1)), X])

        # Ordinary least squares solution
        w = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

        self.intercept_ = float(w[0])
        self.coef_ = w[1:].flatten()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model has not been fitted yet. Call `fit` first.")

        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R^2 score on the provided data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        r2 : float
            R^2 score.
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot
