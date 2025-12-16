
# Here are the tests you can run to check the functionality of the  Principal Component Analysis model

# tests for the obesity dataset used in this model can be found in the test_preprocessing.py file

# packages
import pytest
import numpy as np
import pandas as pd


# load data
# set data path 
repo_root = Path("/Users/doriolson/Desktop/repos/CMOR_438_Final_Repository")
data_path = Path("../../../Data/unsupervised_ObesityDataSet_raw_and_data_sinthetic.csv")

# load dataset
df = pd.read_csv(data_path)

# functions from src
from rice_ml.processing.preprocessing import (get_features,
                                              get_feature_types,
                                              create_preprocessor)
from rice_ml.unsupervised_learning.pca import (train_pca,
                                               plot_explained_variance,
                                               get_pca_loadings,
                                               plot_pca_by_label)


# test that the model produces components
def test_train_pca(df):
    # Train PCA for a small number of components
    pca_model, X_pca = train_pca(df, n_components=2)
    
    assert pca_model is not None
    assert X_pca.shape[0] == df.shape[0]
    assert X_pca.shape[1] == 2

test_train_pca(df)

# test the variance visualization
def test_plot_explained_variance(df):
    try:
        plot_explained_variance(df, max_components=5)
    except Exception as e:
        pytest.fail(f"Explained variance plot failed: {e}")

test_plot_explained_variance(df)

# test that the loadings are created and returned properly
def test_get_pca_loadings(df):
    pca_model, _ = train_pca(df, n_components=3)
    
    loadings = get_pca_loadings(pca_model, df.columns)
    
    assert isinstance(loadings, pd.DataFrame)
    assert loadings.shape[0] == df.shape[1]  # rows = features
    assert loadings.shape[1] == 3  # columns = components

test_get_pca_loadings(df)

# test that the labeled plot worked
def test_plot_pca_by_label(df):
    if "NObeyesdad" in df.columns:
        labels = df["NObeyesdad"]
    else:
        labels = np.random.choice(["A","B"], size=df.shape[0])
    
    pca_model, X_pca = train_pca(df, n_components=2)
    
    try:
        plot_pca_by_label(X_pca, labels)
    except Exception as e:
        pytest.fail(f"PCA by label plot failed: {e}")

test_plot_pca_by_label(df) 