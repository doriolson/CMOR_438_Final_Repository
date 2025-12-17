"""
Unit tests for the custom rice_ml.supervised_learning.decision_trees module.
"""
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

# Import the custom Decision Tree components
from rice_ml.supervised_learning.decision_trees import (
    DecisionTreeClassifier,
    _gini_impurity,
    _entropy,
    _best_split
)

# Tolerance for floating point comparisons
TOL = 1e-6

# ----------------------------- Test Data -----------------------------

# Simple classification data
X_simple = np.array([
    [1.0, 5.0],  # 0
    [2.0, 1.0],  # 0
    [3.0, 6.0],  # 0
    [6.0, 5.0],  # 1
    [7.0, 1.0],  # 1
    [8.0, 6.0],  # 1
    [5.0, 3.0]   # 0
])
y_simple = np.array([0, 0, 0, 1, 1, 1, 0])

# Perfect split data (split on feature 0 at threshold 4.0)
X_perfect = np.array([[1], [2], [5], [6]])
y_perfect = np.array([0, 0, 1, 1])

# ----------------------------- Helper/Internal Tests -----------------------------

def test_impurity_calculations():
    y_pure = np.array([0, 0, 0])
    y_mixed = np.array([0, 1, 0, 1])
    y_equal = np.array([0, 0, 1, 1])
    
    # Pure
    assert _gini_impurity(y_pure) == 0.0
    assert _entropy(y_pure) == 0.0
    
    # Mixed (0.5 Gini)
    assert _gini_impurity(y_mixed) == 0.5
    
    # Equal split
    assert_almost_equal(_gini_impurity(y_equal), 0.5)
    assert_almost_equal(_entropy(y_equal), 1.0)

def test_best_split_perfect():
    # Should split on col 0, threshold 3.5, maximizing gain (Gini=0.5 -> 0, Gain=0.5)
    best_feature, best_threshold, best_gain = _best_split(X_perfect, y_perfect, 'gini')
    assert best_feature == 0
    assert best_threshold == 3.5
    assert_almost_equal(best_gain, 0.5)

def test_best_split_complex():
    # Test on simple mixed data
    # Gini impurity of root is 1 - (4/7)^2 - (3/7)^2 = 0.48979
    # Check for feature 0 splits: 
    # Threshold 4.0: Left=[1,2,3,5] (4x0, 1x1, Gini=0.32), Right=[6,7,8] (3x1, Gini=0)
    # Gain should be current - (4/7)*0.32 - (3/7)*0 = 0.48979 - 0.1828 = 0.3069 (approx)
    
    best_feature, best_threshold, best_gain = _best_split(X_simple, y_simple, 'gini')
    
    # Expected best split is Feature 0 at threshold 4.0
    assert best_feature == 0 
    assert best_threshold == 4.0
    assert best_gain > 0.3 

def test_best_split_no_split():
    # Data is too small or uniform
    X_one = np.array([[1.0, 2.0]])
    y_one = np.array([0])
    best_feature, best_threshold, best_gain = _best_split(X_one, y_one, 'gini')
    assert best_feature is None
    
# ----------------------------- Decision Tree Tests -----------------------------

@pytest.fixture
def fitted_tree():
    # Use max_depth=2 to ensure a constrained, testable tree
    clf = DecisionTreeClassifier(max_depth=2, min_samples_split=2, criterion='gini', random_state=42)
    return clf.fit(X_simple, y_simple)

def test_fit_attributes(fitted_tree):
    assert fitted_tree.n_features_in_ == 2
    assert_array_equal(fitted_tree.classes_, np.array([0, 1]))
    assert fitted_tree.tree_ is not None
    # Check if importance was calculated (not just all zeros)
    assert fitted_tree.feature_importances_.sum() > 0.0
    
def test_predict_simple(fitted_tree):
    # Test a point that should go left (class 0)
    y_pred_left = fitted_tree.predict([[1.0, 1.0]])
    # Test a point that should go right (class 1)
    y_pred_right = fitted_tree.predict([[9.0, 9.0]])
    
    assert_array_equal(y_pred_left, np.array([0]))
    assert_array_equal(y_pred_right, np.array([1]))

def test_score_perfect_data():
    clf = DecisionTreeClassifier(min_samples_split=2).fit(X_perfect, y_perfect)
    assert clf.score(X_perfect, y_perfect) == 1.0

def test_pruning_max_depth():
    # Depth 1: only root split. Predicts majority of children.
    clf_d1 = DecisionTreeClassifier(max_depth=1).fit(X_simple, y_simple)
    # Root split: Feat 0, Threshold 4.0. Left (4x0, 1x1), Right (3x1)
    # Left node majority: 0. Right node majority: 1.
    
    # Query: 5.0 goes Right -> class 1
    pred_right = clf_d1.predict([[5.0, 3.0]])[0]
    assert pred_right == 1
    
    # Query: 3.0 goes Left -> class 0
    pred_left = clf_d1.predict([[3.0, 6.0]])[0]
    assert pred_left == 0

def test_pruning_min_samples_split():
    # min_samples_split=10 should result in a root node only leaf
    clf = DecisionTreeClassifier(min_samples_split=10).fit(X_simple, y_simple)
    # Root node has 7 samples. Cannot split. Should be a leaf.
    assert clf.tree_.is_leaf()
    
    # Predicts the overall majority class (mode of y_simple: 0)
    assert clf.predict([[10, 10]])[0] == 0

def test_pruning_min_samples_leaf():
    # Use data with an equal split on the second layer to test leaf constraint
    X_t = np.array([[1], [2], [3], [4], [5], [6]])
    y_t = np.array([0, 0, 0, 1, 1, 1])
    
    # Root split: Feat 0, Threshold 3.5. Left (3 samples), Right (3 samples).
    # min_samples_leaf=4 means neither branch can form, resulting in a leaf root.
    clf_leaf = DecisionTreeClassifier(min_samples_leaf=4).fit(X_t, y_t)
    assert clf_leaf.tree_.is_leaf()
    
    # Predicts the overall majority class (0 or 1, should be 0 since first in bincount)
    assert clf_leaf.predict([[10]])[0] == 0 

def test_unfitted_call():
    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(X_simple[:1])

def test_dimension_mismatch(fitted_tree):
    X_wrong_dim = np.array([[1.0]])
    with pytest.raises(ValueError, match="features, expected 2"):
        fitted_tree.predict(X_wrong_dim)
        