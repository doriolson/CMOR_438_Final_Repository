"""
Unit tests for DBSCANCommunityDetector in src/rice_ml/unsupervised_learning/dbscan.py.

Uses synthetic data to verify core functionality (clustering, noise, core points).
"""
import numpy as np
import pytest
from rice_ml.unsupervised_learning.dbscan import DBSCANCommunityDetector
from rice_ml.processing.preprocessing import standardize, minmax_scale # Use external utilities for preparation

# Set up test data
@pytest.fixture
def synthetic_data_3clusters():
    """Create data with three distinct, dense clusters and some noise."""
    rng = np.random.default_rng(42)
    
    # Cluster 1: dense, small variance
    C1 = rng.multivariate_normal([0, 0], [[0.05, 0], [0, 0.05]], size=50)
    # Cluster 2: dense, slightly larger variance
    C2 = rng.multivariate_normal([5, 5], [[0.1, 0], [0, 0.1]], size=60)
    # Cluster 3: dense, close to C2
    C3 = rng.multivariate_normal([5.5, 0], [[0.08, 0], [0, 0.08]], size=70)
    # Noise: scattered points
    Noise = rng.uniform(low=[-2, -2], high=[7, 7], size=(20, 2))
    
    X = np.vstack([C1, C2, C3, Noise])
    return X


def test_dbscan_basic_clustering(synthetic_data_3clusters):
    """Test if DBSCAN detects the three main clusters and assigns noise."""
    X = synthetic_data_3clusters
    X_scaled = standardize(X)
    
    # Parameters known to work well for this synthetic data
    eps = 0.35
    min_samples = 10
    
    dbscan = DBSCANCommunityDetector(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    # Check number of unique clusters (excluding noise -1)
    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) == 3, f"Expected 3 clusters, got {len(unique_clusters)}"
    
    # Check for noise points
    noise_count = np.sum(labels == -1)
    assert noise_count > 0, "Expected some points to be labeled as noise (-1)"
    
    # Check cluster sizes (they should be roughly 50, 60, 70)
    cluster_counts = np.unique(labels[labels != -1], return_counts=True)[1]
    expected_counts = [50, 60, 70]
    # Check if the counts are close to expected (allow some noise assignment)
    assert all(count >= (exp - 5) for count, exp in zip(sorted(cluster_counts), sorted(expected_counts))), \
        f"Cluster sizes mismatch: {sorted(cluster_counts)} vs expected ~{sorted(expected_counts)}"


def test_dbscan_min_samples_effect():
    """Test high min_samples results in more noise and fewer clusters."""
    X = np.array([[0,0],[0,1],[10,10],[10,11],[50,50]])
    
    # Case 1: Low min_samples, two small clusters
    dbscan_low = DBSCANCommunityDetector(eps=1.5, min_samples=2)
    labels_low = dbscan_low.fit_predict(X)
    assert len(set(labels_low) - {-1}) == 2, "Should find 2 clusters"
    assert np.sum(labels_low == -1) == 1, "Should find 1 noise point (50,50)"
    
    # Case 2: High min_samples, everything is noise
    dbscan_high = DBSCANCommunityDetector(eps=1.5, min_samples=5)
    labels_high = dbscan_high.fit_predict(X)
    assert len(set(labels_high) - {-1}) == 0, "Should find 0 clusters"
    assert np.all(labels_high == -1), "Everything should be noise"


def test_dbscan_eps_effect():
    """Test low eps results in more noise, high eps results in one cluster."""
    X = np.array([[0,0],[0,1.5],[10,10],[10,11.5]])
    
    # Case 1: Low eps (0.5), small clusters but separated
    dbscan_low_eps = DBSCANCommunityDetector(eps=0.5, min_samples=2)
    labels_low_eps = dbscan_low_eps.fit_predict(X)
    assert len(set(labels_low_eps) - {-1}) == 2, "Should find 2 clusters"
    
    # Case 2: High eps (15.0), all points form one cluster
    dbscan_high_eps = DBSCANCommunityDetector(eps=15.0, min_samples=2)
    labels_high_eps = dbscan_high_eps.fit_predict(X)
    assert len(set(labels_high_eps) - {-1}) == 1, "Should find 1 cluster"
    assert np.sum(labels_high_eps == 0) == 4, "All 4 points should be in the single cluster"


def test_dbscan_pca_utility(synthetic_data_3clusters):
    """Test the integrated PCA transformation utility."""
    X = synthetic_data_3clusters
    X_scaled = standardize(X) # PCA assumes centered data
    
    dbscan = DBSCANCommunityDetector(eps=0.35, min_samples=10).fit(X_scaled)
    
    # Test 2 components
    X_pca, evr = dbscan.pca_transform(n_components=2)
    assert X_pca.shape == (X.shape[0], 2)
    assert evr.shape == (2,)
    assert np.sum(evr) > 0.9 # Should retain most variance for 2D synthetic data
    
    # Test 1 component
    X_pca_1, evr_1 = dbscan.pca_transform(n_components=1)
    assert X_pca_1.shape == (X.shape[0], 1)
    
    # Test error handling
    with pytest.raises(ValueError):
        dbscan.pca_transform(n_components=3) # Only 2 features in X