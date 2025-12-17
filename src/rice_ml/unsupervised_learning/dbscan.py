"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
implementation using only NumPy.

This module provides the DBSCAN algorithm for density-based clustering.

The core implementation uses the standard DBSCAN definitions:
- Epsilon ($\epsilon$ or `eps`): The maximum distance between two samples
  for one to be considered as in the neighborhood of the other.
- Minimum Samples (`min_samples`): The number of samples (or total weight)
  in a neighborhood for a point to be considered as a core point.

Class
-----
DBSCANCommunityDetector
    A pure NumPy implementation of the DBSCAN clustering algorithm.
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple, Union, Sequence

import numpy as np

# Assuming ArrayLike is defined in a shared context, redefining for standalone clarity.
ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ----------------------------- Helpers & Validation -----------------------------

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric ndarray of dtype float (copied from knn.py)."""
    arr = np.asarray(X)
    if arr.ndim != 2:
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

def _euclidean_distances(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of XA and XB.

    Parameters
    ----------
    XA, XB : ndarray, shape (n_a, d), (n_b, d)
        Input matrices.

    Returns
    -------
    D : ndarray, shape (n_a, n_b)
        Euclidean distances.
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b -> then sqrt
    aa = np.sum(XA * XA, axis=1, keepdims=True)       # (n_a, 1)
    bb = np.sum(XB * XB, axis=1, keepdims=True).T     # (1, n_b)
    # numerical stability
    D2 = np.maximum(aa + bb - 2.0 * XA @ XB.T, 0.0)
    return np.sqrt(D2, dtype=float)

def _pca_transform(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA transformation on a data matrix X (NumPy-only).

    Assumes X is already centered (e.g., standardized).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data, typically standardized.
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    X_pca : ndarray, shape (n_samples, n_components)
        Transformed data.
    explained_variance : ndarray, shape (n_components,)
        Variance explained by each component.
    components : ndarray, shape (n_components, n_features)
        The principal axes in feature space.
    """
    if n_components <= 0 or n_components > X.shape[1]:
        raise ValueError("n_components must be positive and less than or equal to n_features.")

    # 1. Covariance matrix (or use SVD on X directly for better stability/speed)
    # Using SVD: X = U S V^T, where V^T contains the principal components
    # The eigenvalues are S^2 / (n-1)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Components (principal axes) are the rows of V^T
    components = Vt[:n_components]
    
    # Explained variance calculation (using eigenvalues)
    n_samples = X.shape[0]
    eigenvalues = S**2 / (n_samples - 1)
    explained_variance = eigenvalues[:n_components]

    # Transformed data: X_pca = X @ components.T
    X_pca = U[:, :n_components] * S[:n_components]
    
    return X_pca, explained_variance, components


# ---------------------------------- Class ----------------------------------

class DBSCANCommunityDetector:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a
        point to be considered as a core point. This includes the point itself.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the training set. Noise is labeled as -1.
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        if not (isinstance(eps, (int, float)) and eps > 0):
            raise ValueError("eps must be a positive float.")
        if not (isinstance(min_samples, (int, np.integer)) and min_samples >= 1):
            raise ValueError("min_samples must be a positive integer >= 1.")

        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike) -> "DBSCANCommunityDetector":
        """
        Perform DBSCAN clustering on X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training features. Must be standardized or scaled.

        Returns
        -------
        self : object
        """
        X_arr = _ensure_2d_float(X, "X")
        n_samples = X_arr.shape[0]

        if n_samples < self.min_samples:
             # Cannot form a core point, everything is noise.
            self.labels_ = np.full(n_samples, -1, dtype=int)
            self.core_sample_indices_ = np.array([], dtype=int)
            self._X = X_arr
            return self

        # 1. Compute pairwise distances
        # We only need distances up to eps, so a dense matrix is okay here.
        D = _euclidean_distances(X_arr, X_arr)
        
        # 2. Find neighbors for each point
        # Neighbors are points where D[i, j] <= eps
        is_neighbor = (D <= self.eps)
        
        # 3. Identify Core Points
        # Core points are those with at least min_samples neighbors (including self)
        n_neighbors = np.sum(is_neighbor, axis=1)
        is_core = (n_neighbors >= self.min_samples)
        self.core_sample_indices_ = np.flatnonzero(is_core)

        # 4. Initialize labels: -1 for noise, 0 for border/unvisited
        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        # 5. Iteratively expand clusters from unvisited core points
        for i in range(n_samples):
            if not is_core[i] or labels[i] != -1:
                # Skip non-core points or already visited points
                continue

            # Start a new cluster
            labels[i] = cluster_id
            
            # Queue for BFS/DFS expansion
            queue = [i]
            while queue:
                p = queue.pop(0)
                
                # Neighbors of core point p
                neighbors_p_idx = np.flatnonzero(is_neighbor[p])

                for q in neighbors_p_idx:
                    if labels[q] == -1:
                        # q is noise or unvisited. Assign it to current cluster.
                        labels[q] = cluster_id
                        
                        # If q is also a core point, add its neighbors to the queue
                        if is_core[q]:
                            queue.append(q)
            
            # Move to the next cluster ID
            cluster_id += 1

        self.labels_ = labels
        self._X = X_arr
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Perform DBSCAN clustering on X and return cluster labels.
        """
        return self.fit(X).labels_
    
    # --- Custom Utility Methods for Analysis (required by strict instruction) ---

    def pca_transform(self, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies PCA for dimensionality reduction, typically for visualization.
        
        NOTE: This utility assumes the data X provided to `fit` was standardized/centered.

        Parameters
        ----------
        n_components : int, default=2
            The number of principal components to keep.

        Returns
        -------
        X_pca : ndarray, shape (n_samples, n_components)
            The data projected onto the principal components.
        explained_variance_ratio : ndarray, shape (n_components,)
            The fraction of variance explained by each component.
        """
        if self._X is None:
            raise RuntimeError("Model is not fitted. Call fit(X) first.")
        
        X_pca, explained_variance, _ = _pca_transform(self._X, n_components)
        
        # Calculate explained variance ratio
        total_variance = np.sum(np.linalg.svd(self._X, full_matrices=False, compute_uv=False)**2) / (self._X.shape[0] - 1)
        explained_variance_ratio = explained_variance / total_variance
        
        return X_pca, explained_variance_ratio