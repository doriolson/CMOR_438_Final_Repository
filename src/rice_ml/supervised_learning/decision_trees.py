"""
Decision Tree Classifier (NumPy-only).

This module provides a simple, dependency-free implementation of the 
Classification and Regression Tree (CART) algorithm for classification. 
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple, Union, Sequence

import numpy as np

__all__ = [
    'DecisionTreeClassifier',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ----------------------------- Helpers & Validation -----------------------------

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric ndarray of dtype float."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    # Ensure all elements are treated as floats for comparison
    return arr.astype(float, copy=False)


def _ensure_1d_int(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Ensure y is a 1D integer ndarray (for encoded labels)."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr.astype(int, copy=False)


def _gini_impurity(y_encoded: np.ndarray) -> float:
    """Calculate Gini impurity for an array of encoded labels (integers)."""
    n_samples = y_encoded.size
    if n_samples == 0:
        return 0.0
    
    # Use bincount for efficiency on integer labels
    counts = np.bincount(y_encoded)
    probabilities = counts / n_samples
    return float(1.0 - np.sum(probabilities**2))


def _entropy(y_encoded: np.ndarray) -> float:
    """Calculate Entropy for an array of encoded labels (integers)."""
    n_samples = y_encoded.size
    if n_samples == 0:
        return 0.0
    
    counts = np.bincount(y_encoded)
    probabilities = counts / n_samples
    
    # Filter out zeros to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _impurity(y_encoded: np.ndarray, criterion: Literal["gini", "entropy"]) -> float:
    """Wrapper for impurity calculation."""
    if criterion == "gini":
        return _gini_impurity(y_encoded)
    elif criterion == "entropy":
        return _entropy(y_encoded)
    raise ValueError(f"Unknown criterion: {criterion}")


def _best_split(
    X: np.ndarray, 
    y_encoded: np.ndarray, 
    criterion: Literal["gini", "entropy"]
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Find the best feature and threshold to split the data.
    """
    m, n = X.shape
    if m <= 1:
        return None, None, None

    current_impurity = _impurity(y_encoded, criterion)
    best_gain = -1e-6  # Ensure initial best gain is slightly negative to accept first split
    best_feature_idx = None
    best_threshold = None

    for col in range(n):
        X_col = X[:, col]
        
        # 1. Identify unique values in this column
        values = np.unique(X_col)
        
        # 2. Define split candidate thresholds (midpoints)
        if len(values) <= 1:
            continue
        
        # Midpoints between unique values
        threshold_candidates = (values[:-1] + values[1:]) / 2.0
        
        # Critical for binary OHE features: ensure 0.5 is checked
        if np.max(values) <= 1.0 and np.min(values) >= 0.0 and 0.5 not in threshold_candidates:
            threshold_candidates = np.append(threshold_candidates, 0.5)

        # Iterate through all viable thresholds for this feature
        for threshold in threshold_candidates:
            # Split data
            left_mask = X_col <= threshold
            y_left = y_encoded[left_mask]
            y_right = y_encoded[~left_mask]

            if y_left.size == 0 or y_right.size == 0:
                continue

            # Calculate gain
            impurity_left = _impurity(y_left, criterion)
            impurity_right = _impurity(y_right, criterion)
            
            p_left = y_left.size / m
            
            # Weighted average of children's impurity
            new_impurity = p_left * impurity_left + (1 - p_left) * impurity_right
            gain = current_impurity - new_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature_idx = col
                best_threshold = threshold
                
    # Only return a split if the best gain is positive enough to matter
    if best_gain > 1e-6:
        return best_feature_idx, best_threshold, best_gain
    else:
        return None, None, None


# ---------------------------------- Node Class ----------------------------------

class _Node:
    """A node in the decision tree."""
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional[_Node] = None,
        right: Optional[_Node] = None,
        value: Optional[np.ndarray] = None, # Stores the encoded class labels of samples in this node
        depth: int = 0
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.depth = depth

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return self.value is not None
    
    def get_prediction(self) -> np.generic:
        """Return the predicted class label (the mode of the value array, which contains encoded labels)."""
        if self.value is None or self.value.size == 0:
            # Fallback (shouldn't happen in a complete tree)
            return np.array([0])[0] # Default to 0 if empty
        
        # Calculate mode: self.value contains integer class IDs
        # np.bincount handles this efficiently
        counts = np.bincount(self.value)
        return np.argmax(counts)


# ---------------------------------- Classifier ----------------------------------

class DecisionTreeClassifier:
    """
    A simple Decision Tree Classifier using the CART algorithm.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: Literal["gini", "entropy"] = "gini",
        random_state: Optional[int] = None,
    ):
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state

        self.tree_: Optional[_Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        
    def _build_tree(
        self, 
        X: np.ndarray, 
        y_encoded: np.ndarray, 
        current_depth: int
    ) -> _Node:
        """Recursively build the decision tree."""
        n_samples, _ = X.shape
        
        # 1. Check for termination conditions (Leaf Node)
        # Condition A: Purity (all labels are the same)
        if np.unique(y_encoded).size == 1:
            return _Node(value=y_encoded, depth=current_depth)

        # Condition B: Max depth reached
        if self.max_depth is not None and current_depth >= self.max_depth:
            return _Node(value=y_encoded, depth=current_depth)

        # Condition C: Min samples split constraint
        if n_samples < self.min_samples_split:
            return _Node(value=y_encoded, depth=current_depth)
        
        # 2. Find the best split
        best_feature_idx, best_threshold, best_gain = _best_split(X, y_encoded, self.criterion)
        
        # Condition D: No beneficial split found (or gain too small)
        if best_gain is None:
            return _Node(value=y_encoded, depth=current_depth)
        
        # Update feature importance (normalized by total samples at the root)
        if self.feature_importances_ is not None:
            # Use gain * (samples at node) to credit this split's importance
            self.feature_importances_[best_feature_idx] += best_gain * n_samples
        
        # 3. Apply split
        left_mask = X[:, best_feature_idx] <= best_threshold
        X_left, y_left = X[left_mask], y_encoded[left_mask]
        X_right, y_right = X[~left_mask], y_encoded[~left_mask]

        # Condition E: Min samples leaf constraint
        if X_left.shape[0] < self.min_samples_leaf or X_right.shape[0] < self.min_samples_leaf:
            # If the split is valid by gain but violates leaf size, treat as a leaf
            return _Node(value=y_encoded, depth=current_depth)

        # 4. Recurse
        left_child = self._build_tree(X_left, y_left, current_depth + 1)
        right_child = self._build_tree(X_right, y_right, current_depth + 1)

        return _Node(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            depth=current_depth
        )


    def fit(self, X: ArrayLike, y: ArrayLike) -> "DecisionTreeClassifier":
        """
        Build a decision tree classifier from the training set (X, y).
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_int(y, "y") # Now strictly enforcing integer labels (0 or 1)
        
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError("X and y length mismatch.")

        # Encode target variable: this is crucial for the integer-based impurity metrics
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        
        self.n_features_in_ = X_arr.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        
        # Start the recursive build
        self.tree_ = self._build_tree(X_arr, y_encoded, current_depth=0)
        
        # Normalize feature importance
        total_importance = self.feature_importances_.sum()
        if total_importance > 0:
            # Normalize by the total importance across all splits in the tree
            self.feature_importances_ /= total_importance
            
        return self

    def _traverse_tree(self, x: np.ndarray) -> np.generic:
        """Traverse the fitted tree to find the prediction for a single sample x."""
        node = self.tree_
        if node is None:
            raise RuntimeError("Model is not fitted.")
            
        while not node.is_leaf():
            feature_idx = node.feature_idx
            threshold = node.threshold
            
            # Safety check: if a non-leaf node has no split info, break and predict majority
            if feature_idx is None or threshold is None:
                break 
                
            if x[feature_idx] <= threshold:
                node = node.left
            else:
                node = node.right
                
            if node is None:
                # Should not happen if the tree is built correctly
                break 

        # Get the encoded prediction (e.g., 0 or 1)
        encoded_pred = node.get_prediction()
        # Map back to the original label (e.g., <=50K or >50K, though here it's 0 or 1)
        return self.classes_[encoded_pred]


    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class labels for samples in X.
        """
        if self.tree_ is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        
        Xq = _ensure_2d_float(X, "X")
        if Xq.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {self.n_features_in_}.")

        # Apply _traverse_tree to each row
        predictions = np.array([self._traverse_tree(x) for x in Xq])
        return predictions

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Classification accuracy on (X, y).
        """
        y_true = _ensure_1d_int(y, "y")
        y_pred = self.predict(X)
        
        if len(y_true) != len(y_pred):
            raise ValueError("X and y lengths do not match.")
            
        return float(np.mean(y_true == y_pred))