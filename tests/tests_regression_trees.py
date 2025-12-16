
# Here are some tests that can be performed on the regression tree model

# packages
import numpy as np
import pandas as pd
import pytest

from rice_ml.processing.preprocessing import (load_and_prepare_data,
                                              build_preprocessor)
from rice_ml.supervised_learning.regression_trees import (train_regression_tree)

# load data
repo_root = Path("/Users/doriolson/Desktop/repos/CMOR_438_Final_Repository")
data_path = Path("../../../Data/adult.csv")


# load data
df = load_and_prepare_data(data_path)


# dataset tests for the regression tree model can be found in test_preprocessing.py
# Data tests apply similarly across all of the unsupervised models


# test the training model
def test_train_regression_tree():
    X, y = load_and_prepare_data("adult.csv")
    preprocessor = build_preprocessor(X)

    model, X_test, y_test = train_regression_tree(
        X, y, preprocessor, max_depth=5
    )

    assert model is not None
    assert len(X_test) == len(y_test)

test_train_regression_tree


# Test how the algorithm evaluated
def test_evaluate_regression_tree():
    X, y = load_and_prepare_data("adult.csv")
    preprocessor = build_preprocessor(X)

    model, X_test, y_test = train_regression_tree(
        X, y, preprocessor, max_depth=5
    )

    metrics = evaluate_regression_tree(model, X_test, y_test)

    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0
    assert -1.0 <= metrics["r2"] <= 1.0

test_evaluate_regression_tree()

# Test the tree visualization
def test_plot_regression_tree():
    X, y = load_and_prepare_data("adult.csv")
    preprocessor = build_preprocessor(X)

    model, _, _ = train_regression_tree(X, y, preprocessor, max_depth=3)

    try:
        plot_regression_tree(model)
    except Exception as e:
        pytest.fail(f"Tree plot failed: {e}")
        
test_plot_regression_tree()