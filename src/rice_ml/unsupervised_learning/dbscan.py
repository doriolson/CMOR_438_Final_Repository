"""
DBSCAN-based community detection implementation for the rice_ml package.

This module provides a simple, interpretable pipeline for performing
unsupervised clustering using DBSCAN on tabular data.

The steps are:
1. Standardize the feature matrix
2. Fit DBSCAN to the data
3. Return cluster labels (-1 indicates noise)

This is designed for educational clarity and mirrors the structure
used in other supervised and unsupervised modules in the package.

Example
-------
>>> import pandas as pd
>>> from rice_ml.unsupervised_learning.dbscan_detector import DBSCANCommunityDetector
>>>
>>> df = pd.read_csv("Obesity_Dataset.csv")
>>> detector = DBSCANCommunityDetector(eps=0.5, min_samples=5)
>>> labels = detector.fit_predict(df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


@dataclass
class DBSCANCommunityDetector:
    """DBSCAN-based community detection for tabular data.

    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples to be considered neighbors.
    min_samples : int, default=5
        Minimum number of samples in a neighborhood for a point to be considered a core point.
    metric : str, default='euclidean'
        Distance metric for DBSCAN.
    scale : bool, default=True
        Whether to standardize features before clustering.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels for each sample (-1 indicates noise).
    model_ : DBSCAN
        Fitted DBSCAN model.
    """

    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    scale: bool = True

    labels_: Optional[np.ndarray] = None
    model_: Optional[DBSCAN] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame | np.ndarray) -> "DBSCANCommunityDetector":
        """Fit DBSCAN to the data.

        Parameters
        ----------
        X : pandas.DataFrame or np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        self : DBSCANCommunityDetector
            Fitted detector.
        """
        X = self._prepare_data(X)

        # Initialize and fit DBSCAN
        self.model_ = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )
        self.labels_ = self.model_.fit_predict(X)
        return self

    def fit_predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : pandas.DataFrame or np.ndarray
            Input feature matrix.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels (-1 indicates noise).
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
