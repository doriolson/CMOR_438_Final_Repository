# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Note: Preprocessing functions are imported from the preprocessing module
try:
    from rice_ml.processing.preprocessing import (get_features,
                                                  get_feature_types,
                                                  create_preprocessor,
                                                  load_and_prepare_data)
except ImportError:
    pass

################ Train Data #############

# train the model
def train_pca(X_processed, n_components=None):
    """
    Train PCA model.
    If n_components=None, all components are kept.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_processed)

    return pca, X_pca


# variance visualization
def plot_explained_variance(pca):
    """Scree plot of explained variance."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.cumsum(pca.explained_variance_ratio_),
        marker="o"
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.axhline(y=0.80, color="red", linestyle="--", label="80% Variance")
    plt.legend()
    plt.show()


########### Visualizations and Evaluation ##########

# 2 dimensional pca vis
def plot_pca_2d(X_pca):
    """2D PCA scatter plot."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Projection (Unsupervised)")
    plt.show()
    
    
# PCA Colored by Obesity Label (Interpretation Only...not training)
def plot_pca_by_label(X_pca, labels):
    """PCA scatter colored by obesity class (interpretation)."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=labels,
        palette="Set2",
        alpha=0.7
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection Colored by Obesity Category")
    plt.show()


# PCA loadings and feature importance
def get_pca_loadings(pca, feature_names, n_components=2):
    """Return PCA loadings for interpretation."""
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=feature_names
    )
    return loadings


####### Guarded Run Block ########
if __name__ == "__main__":
    # Corrected relative path for direct script execution
    DATA_FILE = "../../../Data/unsupervised_ObesityDataSet_raw_and_data_sinthetic.csv"
    
    try:
        df = pd.read_csv(DATA_FILE)
        
        # Prepare features
        X = get_features(df)
        num_features, cat_features = get_feature_types()

        # Preprocessing
        preprocessor = create_preprocessor(num_features, cat_features)
        X_processed = preprocessor.fit_transform(X)

        # Train PCA
        pca, X_pca = train_pca(X_processed)

        # Visualizations
        plot_explained_variance(pca)

        # PCA feature loadings
        feature_names = (
            num_features +
            list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features))
        )

        loadings = get_pca_loadings(pca, feature_names)
        print("\nTop 10 Loadings for PC1:")
        print(loadings.sort_values("PC1", key=abs, ascending=False).head(10))

        plot_pca_2d(X_pca)
        plot_pca_by_label(X_pca, df["NObeyesdad"])
        
    except FileNotFoundError:
        print(f"Could not find data file at {DATA_FILE} for local test.")


