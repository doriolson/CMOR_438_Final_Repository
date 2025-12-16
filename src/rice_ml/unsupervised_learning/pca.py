# import packages
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns


########## Preprocessing Functions (moved to preprocessing.py) #######

# exclude dependent variable
def get_features(df, target_column="NObeyesdad"):
    """Drop target column for unsupervised learning."""
    return df.drop(columns=[target_column])

# categorize variables
def get_feature_types():
    """Return numerical and categorical feature lists."""
    numerical_features = [
        "Age", "Height", "Weight", "FCVC", "NCP",
        "CH2O", "FAF", "TUE"
    ]

    categorical_features = [
        "Gender", "CALC", "FAVC", "SCC", "SMOKE",
        "family_history_with_overweight", "CAEC", "MTRANS"
    ]

    return numerical_features, categorical_features


# preprocess 
def create_preprocessor(numerical_features, categorical_features):
    """Scaling + one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features)
        ]
    )
    
    
    
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



####### Run the data ########

# load in data
df = pd.read_csv("unsupervised_ObesityDataSet_raw_and_data_sinthetic.csv")

# preprocessing
# Prepare features
X = get_features(df)

# Feature types
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
print(loadings.sort_values("PC1", key=abs, ascending=False).head(10))

plot_pca_2d(X_pca)

# interpretation with labels...somewhat supervised but interesting to note
plot_pca_by_label(X_pca, df["NObeyesdad"])


