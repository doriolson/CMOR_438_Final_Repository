
# Here are some tests to run on the K means clustering algorithm

# packages
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# load in data
repo_root = Path("/Users/doriolson/Desktop/repos/CMOR_438_Final_Repository")
data_path = Path("../../../Data/unsupervised_ObesityDataSet_raw_and_data_sinthetic.csv")

# load dataset
df = pd.read_csv(data_path)

from rice_ml.processing.preprocessing import (get_features,
                                              get_feature_types,
                                              create_preprocessor)
from rice_ml.unsupervised_learning.k_means_clustering import (train_kmeans,
                                                              plot_silhouette_scores,
                                                              evaluate_clustering,
                                                              )


# test that the clusters produce labels
def test_train_kmeans_labels(df):
    # Train K-Means with a reasonable number of clusters
    model, labels = train_kmeans(df, n_clusters=3)
    
    assert model is not None
    assert labels.shape[0] == len(df)
    # Check that all label values are within cluster range
    assert set(labels).issubset(set(range(3)))
    
test_train_kmeans_labels(df)


# test the silhouette analysis visualization
def test_plot_silhouette_scores(df):
    try:
        plot_silhouette_scores(df, max_k=10)
    except Exception as e:
        pytest.fail(f"Silhouette score plot failed: {e}")

test_plot_silhouette_scores(df)


# test the evaluation of clustering
def test_evaluate_clustering(df):
    model, labels = train_kmeans(df, n_clusters=3)
    metrics = evaluate_clustering(df, labels)
    
    assert "inertia" in metrics
    assert "silhouette_score" in metrics
    assert metrics["inertia"] >= 0
    assert -1.0 <= metrics["silhouette_score"] <= 1.0

test_evaluate_clustering(df)

# test cluster vs target, if the target exists
def test_cluster_vs_target(df):
    if "NObeyesdad" in df.columns:
        model, labels = train_kmeans(df, n_clusters=3)
        try:
            cluster_vs_target(df, labels, target="NObeyesdad")
        except Exception as e:
            pytest.fail(f"Cluster vs target plot failed: {e}")

test_cluster_vs_target(df)

