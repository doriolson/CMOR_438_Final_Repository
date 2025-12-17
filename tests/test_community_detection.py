"""
Tests for rice_ml.unsupervised_learning.community_detection
(CommunityDetector class).
"""

import numpy as np
import pytest
from rice_ml.unsupervised_learning.community_detection import CommunityDetector

# --- Fixtures for Synthetic Data ---

@pytest.fixture
def synthetic_data_three_groups():
    """Generate synthetic data with three clear clusters (communities)."""
    rng = np.random.default_rng(42)
    n_samples = 150
    n_per_group = n_samples // 3
    
    # Centers
    c1 = np.array([0, 10])
    c2 = np.array([-10, -5])
    c3 = np.array([10, -5])
    
    # Generate data
    X1 = rng.normal(loc=c1, scale=1.0, size=(n_per_group, 2))
    X2 = rng.normal(loc=c2, scale=1.0, size=(n_per_group, 2))
    X3 = rng.normal(loc=c3, scale=1.0, size=(n_per_group, 2))
    
    X = np.vstack([X1, X2, X3])
    
    # Standardize data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / np.where(X_std == 0, 1, X_std)
    
    return X_scaled

# --- Test Cases ---

def test_fit_predict_shape(synthetic_data_three_groups):
    """Test the output shape of fit_predict."""
    X = synthetic_data_three_groups
    n_samples = X.shape[0]
    n_clusters = 3
    
    detector = CommunityDetector(n_clusters=n_clusters, n_neighbors=10, random_state=42)
    labels = detector.fit_predict(X)
    
    assert labels.shape == (n_samples,)
    assert np.unique(labels).size == n_clusters 

def test_n_neighbors_constraint():
    """Test n_neighbors constraint relative to data size."""
    X = np.random.rand(10, 5) # 10 samples
    
    with pytest.raises(ValueError):
        CommunityDetector(n_clusters=2, n_neighbors=10, random_state=42).fit_predict(X)

    CommunityDetector(n_clusters=2, n_neighbors=5, random_state=42).fit_predict(X)

def test_robustness_on_clean_data(synthetic_data_three_groups):
    """
    Test if the detector finds highly distinct clusters using Silhouette Score.
    Clean data should result in a high Silhouette Score (> 0.7).
    """
    X = synthetic_data_three_groups
    n_clusters = 3
    
    detector = CommunityDetector(n_clusters=n_clusters, n_neighbors=30, random_state=42, gamma=1.0)
    labels = detector.fit_predict(X)

    # 1. Check cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    assert np.min(counts) >= 40 # Assert no degenerate clusters (150 samples / 3 clusters)
    
    # 2. Check evaluation metric (Silhouette Score)
    sil_score = detector.evaluate()
    # For clean, well-separated Gaussian blobs, a high score is expected.
    assert sil_score > 0.7 

def test_evaluate_before_fit():
    """Test that calling evaluate before fit raises a RuntimeError."""
    detector = CommunityDetector(n_clusters=3)
    with pytest.raises(RuntimeError):
        detector.evaluate()