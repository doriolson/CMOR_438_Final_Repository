"""
Pytest module for testing the LinearRegression implementation in rice_ml.
"""

import pytest
import numpy as np
from rice_ml.supervised_learning.linear_regression import LinearRegression

# Use a standard tolerance for floating point comparisons
TOL = 1e-4

# --- Fixtures ---

@pytest.fixture
def ols_data():
    """Simple linear data for testing OLS (alpha=0.0)."""
    # y = 2 * x1 - 3 * x2 + 10
    X = np.array([
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, 2.0],
        [4.0, 3.0],
    ], dtype=float)
    y = np.array([12.0, 11.0, 10.0, 9.0], dtype=float)
    # Expected OLS weights (calculated by hand or external solver):
    # Intercept = 15.0, Coef_1 = -1.0, Coef_2 = 0.0 
    # (If using simple y=2x+10, then weights would be: I=6.0, C=2.0)
    # Re-using the simple linear function to be predictable:
    # y = 2 * x + 5 (using just x1)
    X_simple = np.array([
        [1.0], [2.0], [3.0], [4.0]
    ])
    y_simple = np.array([7.0, 9.0, 11.0, 13.0])
    # Expected: Intercept = 5.0, Coef = 2.0
    return X_simple, y_simple, 5.0, 2.0 # X, y, expected_intercept, expected_coef

@pytest.fixture
def ridge_data():
    """Data where Ridge regularization should shrink coefficients."""
    # Highly correlated features (multi-collinearity)
    X = np.array([
        [1.0, 1.01],
        [2.0, 2.02],
        [3.0, 3.03],
    ])
    y = np.array([10.0, 20.0, 30.0])
    return X, y

# --- Tests ---

def test_initialization():
    """Test LinearRegression class initialization and default values."""
    reg = LinearRegression()
    assert reg.alpha == 0.0
    assert not reg._is_fitted
    
    reg_ridge = LinearRegression(alpha=10.0)
    assert reg_ridge.alpha == 10.0

def test_alpha_validation():
    """Test non-numeric or negative alpha value raises ValueError."""
    with pytest.raises(ValueError, match="alpha must be a non-negative float."):
        LinearRegression(alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be a non-negative float."):
        LinearRegression(alpha="invalid")

def test_ols_fit_and_predict(ols_data):
    """Test OLS (alpha=0.0) fit and prediction accuracy."""
    X, y, expected_i, expected_c = ols_data
    
    reg = LinearRegression(alpha=0.0)
    reg.fit(X, y)
    
    # Check fitted parameters
    assert np.isclose(reg.intercept_, expected_i, atol=TOL)
    assert np.isclose(reg.coef_[0], expected_c, atol=TOL)
    
    # Check prediction
    X_test = np.array([[5.0], [6.0]])
    y_pred = reg.predict(X_test)
    y_expected = np.array([15.0, 17.0])
    assert np.allclose(y_pred, y_expected, atol=TOL)

def test_ridge_fit_effect(ridge_data):
    """Test that ridge regularization shrinks coefficients compared to OLS."""
    X, y = ridge_data
    
    # 1. OLS (alpha=0.0) - Should have large, possibly unstable weights due to correlation
    ols_reg = LinearRegression(alpha=0.0).fit(X, y)
    ols_weights = np.abs(ols_reg.coef_)
    
    # 2. Ridge (alpha=10.0) - Weights should be noticeably smaller
    ridge_reg = LinearRegression(alpha=10.0).fit(X, y)
    ridge_weights = np.abs(ridge_reg.coef_)
    
    # Check if ridge weights are strictly smaller (shrinkage effect)
    assert np.all(ridge_weights < ols_weights)

def test_prediction_before_fit_raises():
    """Test calling predict() before fit() raises a RuntimeError."""
    reg = LinearRegression(alpha=0.0)
    with pytest.raises(RuntimeError, match="Model is not fitted. Call fit"):
        reg.predict(np.array([[1.0]]))

def test_feature_mismatch_raises(ols_data):
    """Test calling predict() with the wrong number of features raises ValueError."""
    X, y, _, _ = ols_data
    reg = LinearRegression(alpha=0.0).fit(X, y)
    
    # X was fitted with 1 feature, predict with 2
    X_wrong = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="model was fitted with 1 features."):
        reg.predict(X_wrong)

def test_r2_score_perfect_fit(ols_data):
    """Test R^2 score for a perfect fit should be 1.0."""
    X, y, _, _ = ols_data
    reg = LinearRegression(alpha=0.0).fit(X, y)
    score = reg.score(X, y)
    assert np.isclose(score, 1.0, atol=TOL)

def test_r2_score_baseline(ols_data):
    """Test R^2 score for a baseline model (mean prediction) should be 0.0 (or near)."""
    X, y, _, _ = ols_data
    # Fit a model to constant data, forcing coefficients to zero (prediction = mean)
    X_const = np.ones_like(X) 
    reg = LinearRegression(alpha=1000.0).fit(X_const, y) # High alpha forces weights to 0
    # In this case, the fit is not exactly the mean, so we manually check a score of ~0.0
    
    # A simple prediction of the mean of y_train should yield R^2 close to 0
    y_pred_mean = np.full_like(y, np.mean(y))
    ss_res = np.sum((y - y_pred_mean) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    # Manual baseline R^2 is 0.0
    assert np.isclose(1.0 - ss_res / ss_tot, 0.0, atol=TOL)
    
    # Test our model's score method
    reg_mean = LinearRegression(alpha=1e9).fit(X, y) # High alpha to force weight toward zero
    score = reg_mean.score(X, y)
    # The score should be close to 0.0 if the model is poor/over-regularized
    assert score < 0.1