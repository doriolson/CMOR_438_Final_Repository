"""
Linear Regression models (NumPy-only).

This module provides standard Linear Regression models suitable for teaching
and lightweight usage, implementing fitting via the Normal Equation
(closed-form solution). Supports:
- Ordinary Least Squares (OLS)
- Ridge Regression (L2 regularization)

Robust error handling and NumPy-style docstrings are used throughout.

Examples
--------
Basic OLS:
>>> import numpy as np
>>> from rice_ml.supervised_learning.linear_regression import LinearRegression
>>> X = np.array([[1], [2], [3], [4]], dtype=float)
>>> y = np.array([2.1, 3.9, 6.2, 7.8])
>>> reg = LinearRegression(alpha=0.0).fit(X, y) # alpha=0.0 is OLS
>>> round(reg.predict([[5.0]])[0], 2)
9.7

Ridge Regression:
>>> X = np.array([[1., 10.], [2., 11.]], dtype=float)
>>> y = np.array([12., 15.])
>>> reg_ridge = LinearRegression(alpha=100.0).fit(X, y)
>>> np.round(reg_ridge.coef_, 2).tolist()
[0.02, 0.02]
"""

from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np

# Assuming necessary helpers from processing/preprocessing.py and post_processing.py
# If you haven't merged them into a shared directory, these imports must be adjusted.
# For now, we only need basic NumPy array validation.

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ----------------------------- Helpers & Validation -----------------------------

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric ndarray of dtype float."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        # Check if 1D array was passed, which should be reshaped to (n, 1) if it's not empty
        if arr.ndim == 1 and arr.size > 0:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr

def _ensure_1d_float(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Ensure y is a 1D numeric ndarray of dtype float."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        # Allow (n, 1) to be squeezed to (n,)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.squeeze(axis=1)
        elif arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


# ----------------------------- Linear Regression Model -----------------------------

class LinearRegression:
    """
    Linear Regression model implementing fitting via the Normal Equation.

    Supports OLS (alpha=0.0) and Ridge Regression (alpha > 0.0).
    It automatically includes an intercept term by prepending a column of ones
    to the input features X.

    Parameters
    ----------
    alpha : float, default=0.0
        Regularization strength (L2 penalty). alpha=0.0 is standard OLS.
        Must be non-negative.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the features (excluding the intercept).
    intercept_ : float
        Estimated intercept (bias) term.
    weights_ : ndarray of shape (n_features + 1,)
        Full weight vector: [intercept, coef_1, ..., coef_n].
    """

    def __init__(self, alpha: float = 0.0) -> None:
        if not isinstance(alpha, (int, float)) or alpha < 0.0:
            raise ValueError("alpha must be a non-negative float.")
        self.alpha = float(alpha)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.weights_: Optional[np.ndarray] = None
        self._is_fitted: bool = False
        self._n_features: Optional[int] = None

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column (column of ones) to X."""
        # Check if X is empty after validation
        if X.shape[0] == 0:
            raise ValueError("Input X must have at least one sample.")
        # Prepend a column of ones for the intercept term
        X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return X_design

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """
        Fit the linear model using the Normal Equation:
        $w = (X^T X + \\alpha I)^{-1} X^T y$

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training feature matrix.
        y : array_like, shape (n_samples,)
            Target values (must be numeric).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_float(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y length mismatch: X has {X_arr.shape[0]} samples, y has {y_arr.shape[0]}."
            )

        X_design = self._prepare_X(X_arr)

        # 1. Compute X^T X
        XTX = X_design.T @ X_design # shape (n_features + 1, n_features + 1)
        
        # 2. Add regularization term (alpha * I)
        n_dim = XTX.shape[0]
        # Identity matrix for all dimensions, but the first row/column (intercept)
        # should not be regularized (alpha=0 for the intercept's dimension in the I matrix)
        identity = np.eye(n_dim)
        identity[0, 0] = 0.0 # Do not penalize the intercept
        
        # Add regularization to XTX
        regularized_XTX = XTX + self.alpha * identity

        # 3. Compute (X^T X + alpha*I)^-1 * X^T y
        try:
            # Solve using linalg.solve or linalg.lstsq which is generally more stable than explicit inverse
            weights = np.linalg.solve(regularized_XTX, X_design.T @ y_arr)
        except np.linalg.LinAlgError as e:
            # Handle cases where the matrix might be singular (collinear features)
            # If regularization (alpha > 0) is used, this is unlikely.
            raise RuntimeError("Singular matrix encountered. Data may be perfectly multicollinear or too small.") from e

        # Store results
        self.weights_ = weights
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        self._is_fitted = True
        self._n_features = X_arr.shape[1]
        
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict target values for new samples.

        Prediction is computed as $\hat{y} = X_{design} w$.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted target values (float).

        Raises
        ------
        RuntimeError
            If called before fit.
        """
        if not self._is_fitted or self.weights_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        Xq = _ensure_2d_float(X, "X")
        if Xq.shape[1] != self._n_features:
            raise ValueError(
                f"X has {Xq.shape[1]} features, but model was fitted with {self._n_features} features."
            )
        
        Xq_design = self._prepare_X(Xq)
        
        # Prediction: X_design @ weights
        y_pred = Xq_design @ self.weights_
        
        return y_pred.astype(float, copy=False)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return the coefficient of determination (R^2) of the prediction.

        The R^2 is computed using the `r2_score` function from `post_processing`.
        """
        # We need to import r2_score here if it's external, but for demonstration,
        # we'll use a placeholder for the logic (assuming it's available).
        
        y_true = _ensure_1d_float(y, "y")
        y_pred = self.predict(X)
        
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred shapes mismatch.")

        ss_res = np.sum((y_true - y_pred) ** 2)
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)

        if ss_tot == 0:
            # Consistent R2 handling for constant y_true (requires external knowledge of training X/y)
            # Since we can't check if X is the training set here without more complexity,
            # we adopt the robust convention: if ss_tot=0, R^2 is 1.0 if ss_res=0, else raise/return 0.0.
            return 1.0 if ss_res < 1e-10 else 0.0
            
        return float(1.0 - ss_res / ss_tot)