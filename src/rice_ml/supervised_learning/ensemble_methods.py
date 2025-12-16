"""
Random Forest (Bagged Decision Trees) classifier for the rice_ml package.

This module implements a simple ensemble method using multiple
decision trees with bootstrap aggregation (bagging). Each tree is grown
using the DecisionTreeClassifier implemented earlier. No scikit-learn
dependency is required.

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.ensemble import RandomForestClassifier
>>> from rice_ml.supervised_learning.decision_tree import DecisionTreeClassifier
>>>
>>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
>>> y = np.array([0, 0, 1, 1])
>>>
>>> rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
>>> rf.fit(X, y)
>>> rf.predict(X)
array([0, 0, 1, 1])
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier

class RandomForestClassifier:
    """Random Forest classifier using bagging of decision trees.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of each individual tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    max_features : int or float, optional
        Number of features to consider when looking for best split in each tree.
    random_state : int or None, optional
        Random seed for reproducibility.

    Attributes
    ----------
    trees_ : list
        List of DecisionTreeClassifier instances.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[float | int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit the random forest on the training data."""
        n_samples = X.shape[0]
        self.trees_ = []

        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = self._rng.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self._rng.integers(0, 1_000_000),
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by majority vote from all trees."""
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        # Majority vote
        y_pred = np.apply_along_axis(lambda x: np.bincount(x, minlength=np.max(x)+1).argmax(), axis=0, arr=predictions)
        return y_pred
