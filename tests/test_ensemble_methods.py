"""
Ensemble methods for classification: RandomForest and VotingClassifier.

This module provides dependency-free implementations of two major ensemble 
techniques, relying on base estimators (like the custom DecisionTreeClassifier)
and NumPy for all core operations.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union, Tuple, List, Callable, Literal, Dict

import numpy as np
from rice_ml.processing.preprocessing import train_test_split # Reusing custom split
from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier # Using the custom DT

__all__ = [
    'RandomForestClassifier',
    'VotingClassifier',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ----------------------------- Helper Functions -----------------------------

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric ndarray of dtype float."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr.astype(float, copy=False)


def _ensure_1d_int(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Ensure y is a 1D integer ndarray (for encoded labels)."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    # Target labels must be convertible to integers for voting/metrics
    return arr.astype(int, copy=False)


def _get_majority_vote(y_preds: np.ndarray) -> np.generic:
    """Find the majority class from a 1D array of predictions."""
    if y_preds.size == 0:
        return np.array([0])[0]
    # Use bincount to find the mode (most frequent item)
    counts = np.bincount(y_preds)
    return np.argmax(counts)


# ----------------------------- Random Forest -----------------------------

class RandomForestClassifier:
    """
    A Random Forest Classifier implementation using bootstrap aggregation (Bagging)
    and random subspace sampling (feature sampling) over multiple Decision Trees.
    
    Relies on the custom DecisionTreeClassifier for base estimation.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Union[str, float] = "sqrt", # 'sqrt', 'log2', or float (0.0, 1.0]
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        criterion: Literal["gini", "entropy"] = "gini",
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.criterion = criterion
        
        # Internal properties
        self.estimators_: List[DecisionTreeClassifier] = []
        self.feature_indices_: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        
        # Random number generator
        self.rng = np.random.default_rng(random_state)

    def _get_max_features(self, n_features: int) -> int:
        """Calculate the number of features to sample based on max_features parameter."""
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
            else:
                raise ValueError(f"Unknown max_features string: {self.max_features}")
        elif isinstance(self.max_features, (float, np.floating)):
            if 0.0 < self.max_features <= 1.0:
                return max(1, int(self.max_features * n_features))
            else:
                raise ValueError("max_features float must be in (0.0, 1.0].")
        else:
            raise TypeError("max_features must be 'sqrt', 'log2', or a float in (0.0, 1.0].")


    def fit(self, X: ArrayLike, y: ArrayLike) -> "RandomForestClassifier":
        """
        Build a Random Forest from the training set (X, y).
        
        The fit process trains n_estimators trees, each on a bootstrapped
        sample (if self.bootstrap is True) and with a random subset of features.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_int(y, "y")
        
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError("X and y length mismatch.")

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        self.feature_importances_ = np.zeros(n_features, dtype=float)
        
        n_features_sample = self._get_max_features(n_features)
        feature_pool = np.arange(n_features)

        self.estimators_ = []
        self.feature_indices_ = []

        for i in range(self.n_estimators):
            # 1. Feature Subspace (Random Subspace Sampling)
            # Draw a random subset of features without replacement
            feature_indices = self.rng.choice(
                feature_pool, 
                size=n_features_sample, 
                replace=False
            )
            self.feature_indices_.append(feature_indices)
            
            # Select the feature subset
            X_sub = X_arr[:, feature_indices]
            
            # 2. Sample Aggregation (Bagging/Bootstrap)
            if self.bootstrap:
                # Draw samples with replacement
                sample_indices = self.rng.choice(
                    n_samples, 
                    size=n_samples, 
                    replace=True
                )
                X_bag, y_bag = X_sub[sample_indices], y_encoded[sample_indices]
            else:
                X_bag, y_bag = X_sub, y_encoded

            # 3. Train Base Estimator (Decision Tree)
            # Note: We must pass the *encoded* y_bag (0, 1, 2...) for the custom DT,
            # but we use the un-encoded class labels for the overall forest prediction mapping.
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                random_state=self.rng.integers(1, 10000) # Use new seed for each tree
            )
            # Pass the encoded y to the custom DT
            tree.fit(X_bag, y_bag)
            self.estimators_.append(tree)
            
            # 4. Aggregate Feature Importance
            if tree.feature_importances_ is not None:
                # Map local importance back to the global feature index
                for f_local, f_global in enumerate(feature_indices):
                    # Weight by the number of samples at the root of the tree (n_samples)
                    self.feature_importances_[f_global] += (
                        tree.feature_importances_[f_local]
                    )

        # Normalize total importance by the number of trees
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= self.n_estimators
            
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class labels for samples in X using majority voting across trees.
        """
        if not self.estimators_:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        
        Xq = _ensure_2d_float(X, "X")
        if Xq.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {self.n_features_in_}.")

        # Collect predictions from all trees
        all_preds = []
        for tree, feature_indices in zip(self.estimators_, self.feature_indices_):
            # Use only the subset of features this tree was trained on
            X_sub = Xq[:, feature_indices]
            
            # The custom DT.predict returns the *original* class label (e.g., 0 or 1)
            # We must map these back to their integer indices (0 or 1) for voting
            tree_preds = tree.predict(X_sub)
            
            # Map original labels back to encoded indices (0, 1, 2...)
            # We use the internal self.classes_ from the fit process
            map_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
            
            # Use the map to get the integer index for voting
            encoded_preds = np.array([map_to_idx.get(p) for p in tree_preds], dtype=int)
            all_preds.append(encoded_preds)
        
        # Shape: (n_estimators, n_samples)
        all_preds_arr = np.array(all_preds).T 

        # Aggregate predictions using majority voting (on the encoded indices)
        final_encoded_preds = np.apply_along_axis(_get_majority_vote, axis=1, arr=all_preds_arr)
        
        # Map the final encoded prediction indices back to the original class labels
        final_preds = self.classes_[final_encoded_preds]
        
        return final_preds

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Classification accuracy on (X, y)."""
        y_true = _ensure_1d_int(y, "y")
        y_pred = self.predict(X)
        
        if len(y_true) != len(y_pred):
            raise ValueError("X and y lengths do not match.")
            
        return float(np.mean(y_true == y_pred))


# ----------------------------- Voting Classifier -----------------------------

class VotingClassifier:
    """
    A Voting Classifier implementation for classification, combining predictions
    from multiple diverse estimators using majority vote (hard) or weighted 
    average of probabilities (soft).
    
    This class is an ensemble meta-estimator.
    """
    
    def __init__(
        self,
        estimators: List[Tuple[str, object]], # List of (name, estimator) tuples
        voting: Literal["hard", "soft"] = "hard",
        weights: Optional[Sequence[float]] = None,
    ):
        if not estimators:
            raise ValueError("Estimators list cannot be empty.")
        if voting not in ["hard", "soft"]:
            raise ValueError("Voting must be 'hard' or 'soft'.")
            
        self.estimators = estimators
        self.voting = voting
        self.weights = np.asarray(weights, dtype=float) if weights is not None else None
        
        # Internal properties
        self.fitted_estimators_: List[object] = []
        self.classes_: Optional[np.ndarray] = None
        
    def fit(self, X: ArrayLike, y: ArrayLike) -> "VotingClassifier":
        """Fit all individual estimators."""
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_int(y, "y")
        
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError("X and y length mismatch.")
            
        self.classes_, _ = np.unique(y_arr, return_inverse=True)
        self.fitted_estimators_ = []

        # Fit each estimator
        for name, estimator in self.estimators:
            # We assume estimators have a .fit(X, y) method
            estimator.fit(X_arr, y_arr)
            self.fitted_estimators_.append(estimator)
            
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class labels for samples in X using the specified voting strategy.
        """
        if not self.fitted_estimators_:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
            
        Xq = _ensure_2d_float(X, "X")
        n_samples = Xq.shape[0]

        if self.voting == "hard":
            # Hard Voting: Majority of predicted labels
            
            # predictions_list: List of (n_samples,) arrays of original labels
            predictions_list = [
                estimator.predict(Xq) for estimator in self.fitted_estimators_
            ]
            
            # Map original labels (e.g., 0, 1) back to integer indices (0, 1) for voting
            map_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
            
            # encoded_preds: (n_estimators, n_samples) array of integer indices (0, 1, 2...)
            encoded_preds = np.array([
                [map_to_idx.get(p) for p in preds] 
                for preds in predictions_list
            ], dtype=int).T 

            # final_encoded_preds: (n_samples,) array of majority class indices
            final_encoded_preds = np.apply_along_axis(_get_majority_vote, axis=1, arr=encoded_preds)
            
            # Map back to original labels
            return self.classes_[final_encoded_preds]

        elif self.voting == "soft":
            # Soft Voting: Weighted average of class probabilities
            
            # Ensure all estimators support predict_proba
            if not all(hasattr(e, 'predict_proba') for e in self.fitted_estimators_):
                raise AttributeError("Soft voting requires all estimators to have a 'predict_proba' method.")
                
            # all_probas: (n_estimators, n_samples, n_classes)
            all_probas = []
            for estimator in self.fitted_estimators_:
                proba = estimator.predict_proba(Xq)
                # Simple validation of proba shape
                if proba.ndim != 2 or proba.shape[1] != self.classes_.size:
                    raise ValueError(f"Estimator {estimator} returned probability array of shape {proba.shape}, expected (n_samples, n_classes).")
                all_probas.append(proba)
            
            all_probas_arr = np.array(all_probas) # (n_estimators, n_samples, n_classes)
            
            if self.weights is None:
                weights = np.ones(len(self.estimators), dtype=float)
            elif self.weights.size != len(self.estimators):
                 raise ValueError("Weights must have the same size as the number of estimators.")

            # Weighted average of probabilities
            # Broadcast weights: (n_estimators, 1, 1) * (n_estimators, n_samples, n_classes)
            weighted_probas = all_probas_arr * self.weights[:, None, None]
            
            # Average probabilities across estimators: (n_samples, n_classes)
            avg_probas = np.sum(weighted_probas, axis=0) / np.sum(self.weights)
            
            # Predict the class with the highest average probability (encoded index)
            final_encoded_preds = np.argmax(avg_probas, axis=1)
            
            # Map back to original labels
            return self.classes_[final_encoded_preds]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Classification accuracy on (X, y)."""
        y_true = _ensure_1d_int(y, "y")
        y_pred = self.predict(X)
        
        if len(y_true) != len(y_pred):
            raise ValueError("X and y lengths do not match.")
            
        return float(np.mean(y_true == y_pred))

# Note: The VotingClassifier currently assumes base estimators (like the custom 
# DecisionTreeClassifier or a custom LogisticRegression) have compatible 
# .predict and .predict_proba methods. You will need to ensure your custom 
# LogisticRegression and DecisionTreeClassifier include a predict_proba method 
# for soft voting to work.