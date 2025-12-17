"""
Graph-based Community Detection via Spectral Clustering (NumPy-only).

This module implements a CommunityDetector based on Spectral Clustering,
incorporating all dependencies including K-Means and Silhouette Score
calculation to adhere to strict single-file constraints.

Classes
-------
CommunityDetector
    Performs graph-based clustering (community detection) and evaluation.
"""

from __future__ import annotations
from typing import Literal, Optional, Sequence, Union, Tuple
import numpy as np

# --- Internal Helpers and ArrayLike Definition ---
ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric ndarray of dtype float."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    arr = arr.astype(float, copy=False)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return arr

def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    """Get NumPy random generator from seed."""
    if seed is None:
        return np.random.default_rng()
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(seed))

def _pairwise_euclidean_distances(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    aa = np.sum(XA * XA, axis=1, keepdims=True)
    bb = np.sum(XB * XB, axis=1, keepdims=True).T
    D2 = np.maximum(aa + bb - 2.0 * XA @ XB.T, 0.0)
    return np.sqrt(D2, dtype=float)


# -------------------------- Integrated K-Means Logic --------------------------

class _KMeansInertialClustering:
    """Minimal, pure NumPy K-Means implementation for internal use."""

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf

    def fit(self, X: np.ndarray):
        best_inertia = np.inf
        rng = _rng_from_seed(self.random_state)
        
        for i in range(self.n_init):
            init_seed = rng.integers(0, 2**32 - 1) 
            _, labels, inertia = self._run_single(X, seed=init_seed)
            
            if inertia < best_inertia:
                best_inertia = inertia
                self.labels_ = labels
        
        self.inertia_ = best_inertia
        return self

    def _run_single(self, X: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run a single K-Means initialization."""
        n_samples, n_features = X.shape
        rng = np.random.default_rng(seed)
        
        # Initialization (Random samples)
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centers = X[indices].copy()
        
        current_inertia = np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Assignment
            D = _pairwise_euclidean_distances(X, centers)
            labels = np.argmin(D, axis=1)
            
            new_inertia = np.sum(np.min(D, axis=1)**2)
            
            # M-step: Update
            new_centers = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if cluster_points.shape[0] > 0:
                    new_centers[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centers[k] = X[rng.choice(n_samples)] # Re-initialize empty cluster
            
            # Check for convergence
            center_shift = np.sum((new_centers - centers) ** 2)
            if center_shift <= self.tol * self.tol:
                centers = new_centers
                break
            
            centers = new_centers
            if (current_inertia - new_inertia) < self.tol:
                 break
            current_inertia = new_inertia
        
        # Final assignment and inertia calculation
        D = _pairwise_euclidean_distances(X, centers)
        labels = np.argmin(D, axis=1)
        final_inertia = np.sum(np.min(D, axis=1)**2)
        
        return centers, labels, final_inertia

# -------------------------- Evaluation Metric Logic --------------------------

def _calculate_silhouette_score_internal(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the mean Silhouette Score for a set of data points and labels.
    Used internally by the CommunityDetector.
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        return 0.0
    
    D = _pairwise_euclidean_distances(X, X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1 or n_clusters >= n_samples:
        return 0.0
        
    silhouette_values = np.zeros(n_samples)
    label_masks = {label: (labels == label) for label in unique_labels}
    
    for i in range(n_samples):
        current_label = labels[i]
        
        # --- a(i): Mean intra-cluster distance ---
        same_cluster_dists = D[i, label_masks[current_label]]
        n_c = len(same_cluster_dists)
        if n_c <= 1:
            a_i = 0.0
        else:
            a_i = same_cluster_dists.sum() / (n_c - 1) 
            
        # --- b(i): Minimum mean nearest-cluster distance ---
        b_i = np.inf
        
        for neighbor_label in unique_labels:
            if neighbor_label != current_label:
                neighbor_cluster_dists = D[i, label_masks[neighbor_label]]
                b_neighbor = neighbor_cluster_dists.mean()
                if b_neighbor < b_i:
                    b_i = b_neighbor

        # --- Calculate silhouette coefficient s(i) ---
        if a_i == b_i:
            s_i = 0.0
        elif a_i < b_i:
            s_i = (b_i - a_i) / b_i
        else: # a_i > b_i
            s_i = (b_i - a_i) / a_i
        
        silhouette_values[i] = s_i

    return silhouette_values.mean()


# -------------------------- Community Detector Class --------------------------

class CommunityDetector:
    """
    Community Detection via Spectral Clustering on a k-NN graph.

    This class provides the fit_predict method for clustering and the 
    evaluate method for calculating the Silhouette Score.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of communities/clusters to find.
    n_neighbors : int, default=10
        The number of nearest neighbors to use for graph construction (k in k-NN).
    random_state : int, optional
        Seed for K-Means initialization.
    gamma : float, default=1.0
        Tuning parameter for the similarity (weight) function: exp(-gamma * d^2).
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_neighbors: int = 10,
        random_state: Optional[int] = None,
        gamma: float = 1.0,
    ) -> None:
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        if not (isinstance(gamma, (float, int)) and gamma > 0):
            raise ValueError("gamma must be a positive number.")

        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.gamma = float(gamma)

        self._kmeans_internal: Optional[_KMeansInertialClustering] = None
        self.labels_: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None # Store X for evaluation

    def _build_similarity_matrix(self, X: np.ndarray) -> np.ndarray:
        """Build the k-NN similarity matrix (W)."""
        n_samples = X.shape[0]
        D = _pairwise_euclidean_distances(X, X)
        
        W = np.zeros((n_samples, n_samples), dtype=float)
        
        for i in range(n_samples):
            sorted_indices = np.argsort(D[i, :])
            k_indices = sorted_indices[1:min(self.n_neighbors + 1, n_samples)]
            d_k = D[i, k_indices]
            weights = np.exp(-self.gamma * d_k**2)
            W[i, k_indices] = weights
            
        W = np.maximum(W, W.T) # Symmetrize 
        np.fill_diagonal(W, 0.0)
        return W

    def _spectral_embedding(self, W: np.ndarray) -> np.ndarray:
        """Compute the Spectral Embedding."""
        n_samples = W.shape[0]
        D_diag = np.sum(W, axis=1)
        D = np.diag(D_diag)
        L = D - W # Unnormalized Laplacian
        
        if self.n_clusters >= n_samples:
             raise ValueError("n_clusters must be less than the number of samples.")

        # Compute eigenvectors (eigh for symmetric matrix)
        eigen_vals, eigen_vecs = np.linalg.eigh(L)
        
        # Select the n_clusters smallest eigenvectors (index 0 to n_clusters-1).
        embedding = eigen_vecs[:, :self.n_clusters] 
        
        # Normalize the rows
        row_norms = np.linalg.norm(embedding, axis=1)
        row_norms[row_norms == 0] = 1.0 
        embedding = embedding / row_norms[:, None]
        
        return embedding

    def fit(self, X: ArrayLike):
        """Fit the Community Detector."""
        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[0] <= self.n_neighbors:
             raise ValueError("n_samples must be greater than n_neighbors.")
             
        self._X_fit = X_arr # Store data for evaluation

        # 1. Build Similarity Graph
        W = self._build_similarity_matrix(X_arr)

        # 2. Spectral Embedding
        Y = self._spectral_embedding(W)

        # 3. Cluster embedding with Integrated K-Means
        self._kmeans_internal = _KMeansInertialClustering(
            n_clusters=self.n_clusters,
            n_init=10, 
            random_state=self.random_state
        )
        self._kmeans_internal.fit(Y)
        
        # Final labels
        self.labels_ = self._kmeans_internal.labels_
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """Fit the model and return the community labels."""
        self.fit(X)
        if self.labels_ is None:
            raise RuntimeError("Labels were not generated after fitting.")
        return self.labels_

    def evaluate(self) -> float:
        """
        Calculate the Silhouette Score for the fitted clustering.
        
        Requires X (scaled features) and the labels generated by fit_predict.
        
        Returns
        -------
        float
            The mean Silhouette Score.
        """
        if self._X_fit is None or self.labels_ is None:
            raise RuntimeError("Model must be fitted before calling evaluate().")
            
        return _calculate_silhouette_score_internal(self._X_fit, self.labels_)