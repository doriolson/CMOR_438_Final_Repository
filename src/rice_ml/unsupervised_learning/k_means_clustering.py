# Here is the source code for k means clustering

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Import necessary functions for the execution block
# Note: These are needed if the script is run directly.
try:
    from rice_ml.processing.preprocessing import (get_features,
                                                  get_feature_types,
                                                  create_preprocessor,
                                                  load_and_prepare_data) # Assuming this is the loading function
except ImportError:
    pass 


################# Functions for Model Training ##################

## determine K: 

# elbow curve
def plot_elbow_curve(X_processed, k_range=range(2, 11)):
    """
    Plots inertia values for different k.
    """
    inertia = []

    for k in k_range:
        # NOTE: n_init=10 is the default in newer scikit-learn versions
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_processed)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show() # 

# Silhouette Function Analysis
def plot_silhouette_scores(X_processed, k_range=range(2, 11)):
    """
    Plots silhouette scores for different k.
    """
    scores = []

    for k in k_range:
        # n_init=10 is the default in newer scikit-learn versions
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        score = silhouette_score(X_processed, labels)
        scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, scores, marker="o", color="green")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Analysis")
    plt.show()

## actually train the algorithm
def train_kmeans(preprocessor, X, n_clusters):
    """
    Trains K-Means model using a preprocessing pipeline.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=50
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("kmeans", kmeans)
    ])

    cluster_labels = pipeline.fit_predict(X)

    return pipeline, cluster_labels


############ Evaluation and Visualizations ###########

# evaluations 

def plot_cluster_distribution(df, cluster_column="Cluster"):
    """
    Plots number of samples per cluster.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=cluster_column, data=df)
    plt.title("Cluster Distribution")
    plt.show()


def evaluate_clustering(X_processed, cluster_labels):
    """
    Returns silhouette score.
    """
    score = silhouette_score(X_processed, cluster_labels)
    return score


##  interpret clusters

def cluster_vs_target(df, target_column="NObeyesdad"):
    """
    Crosstab of clusters vs true obesity labels (interpretation only).
    """
    return pd.crosstab(df["Cluster"], df[target_column])

def cluster_numeric_summary(df, numerical_features):
    """
    Mean numerical values per cluster.
    """
    return df.groupby("Cluster")[numerical_features].mean()


## visualizations

def plot_pca_clusters(X_processed, cluster_labels):
    """
    Visualizes clusters in 2D using PCA.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": cluster_labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        data=pca_df
    )
    plt.title("K-Means Clusters (PCA Projection)")
    plt.show() # 


## Guarded execution block ###
if __name__ == "__main__":
    # FIX 1: Corrected relative path for execution from notebook's CWD
    DATA_FILE = "../../../Data/unsupervised_ObesityDataSet_raw_and_data_sinthetic.csv" 
    
    # Load data
    # FIX 2: Assuming load_and_prepare_data is the correct function
    df = load_and_prepare_data(DATA_FILE)

    # Prepare features
    X = get_features(df)

    # Feature types
    num_features, cat_features = get_feature_types()

    # Preprocessing
    preprocessor = create_preprocessor(num_features, cat_features)

    # Transform data once for evaluation plots
    X_processed = preprocessor.fit_transform(X)

    # Model selection
    plot_elbow_curve(X_processed)
    plot_silhouette_scores(X_processed)

    # Train final model
    optimal_k = 4
    pipeline, clusters = train_kmeans(preprocessor, X, optimal_k)

    # Attach clusters
    df["Cluster"] = clusters

    # Evaluation
    sil_score = evaluate_clustering(X_processed, clusters)
    print(f"Silhouette Score: {sil_score:.3f}")

    # Visualization
    plot_cluster_distribution(df)
    plot_pca_clusters(X_processed, clusters)

    # Interpretation
    print(cluster_vs_target(df))
    print(cluster_numeric_summary(df, num_features))