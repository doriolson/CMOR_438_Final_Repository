"""
Community detection implementation for the rice_ml package.

This module provides a simple, interpretable pipeline for performing
community detection on tabular data using a graph-based approach.

The steps are:
1. Build a k-nearest-neighbor graph from feature vectors
2. Convert the graph to a NetworkX graph
3. Detect communities using the Louvain method

This is designed for educational clarity and mirrors the structure
used in other supervised and unsupervised modules in the package.

Example
-------
>>> import pandas as pd
>>> from rice_ml.unsupervised_learning.community_detection import CommunityDetector
>>>
>>> df = pd.read_csv("obesity.csv")
>>> detector = CommunityDetector(n_neighbors=10, random_state=42)
>>> labels = detector.fit_predict(df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

# Louvain community detection
from networkx.algorithms.community import louvain_communities


@dataclass
class CommunityDetector:
    """Graph-based community detection using k-NN + Louvain.

    Parameters
    ----------
    n_neighbors : int, default=10
        Number of neighbors used to build the k-NN graph.
    resolution : float, default=1.0
        Resolution parameter for the Louvain algorithm.
        Higher values lead to more communities.
    random_state : int or None, optional
        Random seed for reproducibility.
    scale : bool, default=True
        Whether to standardize features before building the graph.

    Attributes
    ----------
    labels_ : np.ndarray
        Community label for each sample.
    graph_ : nx.Graph
        Graph constructed from the k-NN adjacency matrix.
    """

    n_neighbors: int = 10
    resolution: float = 1.0
    random_state: Optional[int] = None
    scale: bool = True

    labels_: Optional[np.ndarray] = None
    graph_: Optional[nx.Graph] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame | np.ndarray) -> "CommunityDetector":
        """Fit the community detection model.

        Parameters
        ----------
        X : pandas.DataFrame or np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        self : CommunityDetector
            Fitted detector.
        """
        X = self._prepare_data(X)

        # Build k-NN graph (sparse adjacency matrix)
        knn_graph = kneighbors_graph(
            X,
            n_neighbors=self.n_neighbors,
            mode="connectivity",
            include_self=False
        )

        # Convert to NetworkX graph
        self.graph_ = nx.from_scipy_sparse_array(knn_graph)

        # Run Louvain community detection
        communities = louvain_communities(
            self.graph_,
            resolution=self.resolution,
            seed=self.random_state
        )

        # Convert community sets to label vector
        self.labels_ = self._communities_to_labels(communities, X.shape[0])

        return self

    def fit_predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit the model and return community labels.

        Parameters
        ----------
        X : pandas.DataFrame or np.ndarray
            Input feature matrix.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Community labels.
        """
        self.fit(X)
        return self.labels_

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _prepare_data(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert input to NumPy array and optionally standardize."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features).")

        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        return X

    @staticmethod
    def _communities_to_labels(communities, n_samples: int) -> np.ndarray:
        """Convert Louvain communities to integer labels."""
        labels = np.empty(n_samples, dtype=int)

        for label, community in enumerate(communities):
            for idx in community:
                labels[idx] = label

        return labels
