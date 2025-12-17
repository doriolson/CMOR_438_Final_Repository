"""
Ensemble Methods and Base Learners (NumPy-only).

This module implements:
- DecisionTreeClassifier: Binary/Multiclass classification using Gini impurity.
- LogisticRegression: Binary classification using Gradient Descent.
- VotingClassifier: Soft/Hard voting ensemble logic.
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Optional, Sequence, Union, List, Tuple
from rice_ml.processing.post_processing import accuracy_score

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]

# ---------------------------------------------------------------------
# Base Learners (Required for the Ensemble)
# ---------------------------------------------------------------------

class DecisionTreeClassifier:
    """Simple Decision Tree Classifier using Gini Impurity."""
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = np.bincount(y, minlength=self.n_classes) / n_samples
            return {'leaf': True, 'probs': leaf_value}

        feat_idxs = np.random.choice(n_features, n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        if best_feat is None:
            leaf_value = np.bincount(y, minlength=self.n_classes) / n_samples
            return {'leaf': True, 'probs': leaf_value}

        left_idxs = X[:, best_feat] < best_thresh
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return {'leaf': False, 'feature': best_feat, 'threshold': best_thresh, 'left': left, 'right': right}

    def _best_split(self, X, y, feat_idxs):
        best_gini = 999
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gini = self._gini_impurity(X_column, y, threshold)
                if gini < best_gini:
                    best_gini, split_idx, split_thresh = gini, feat_idx, threshold
        return split_idx, split_thresh

    def _gini_impurity(self, X_column, y, threshold):
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0: return 1
        
        def calculate_gini(labels):
            probs = np.bincount(labels) / len(labels)
            return 1 - np.sum(probs**2)
        
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        return (n_l/n) * calculate_gini(y[left_idxs]) + (n_r/n) * calculate_gini(y[right_idxs])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node['leaf']: return node['probs']
        if x[node['feature']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

class LogisticRegression:
    """Logistic Regression with Gradient Descent."""
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            model_out = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(model_out)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        p1 = self._sigmoid(z)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# ---------------------------------------------------------------------
# Ensemble Logic
# ---------------------------------------------------------------------

class VotingClassifier:
    """
    Soft Voting Ensemble Classifier.
    Combines multiple models by averaging their predicted probabilities.
    """
    def __init__(self, estimators: List[Tuple[str, object]]):
        self.estimators = estimators

    def fit(self, X: np.ndarray, y: np.ndarray):
        for name, model in self.estimators:
            model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_probas = np.array([model.predict_proba(X) for _, model in self.estimators])
        return np.mean(all_probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))