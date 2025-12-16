
# Here are the tests to run for the perceptron and multilayer perceptron models


# load packages
import pytest
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from rice_ml.processing.preprocessing import load_and_prepare_data


repo_root = Path("/Users/doriolson/Desktop/repos/CMOR_438_Final_Repository")
data_path = Path("../../../Data/adult.csv")


# load data
df = load_and_prepare_data(data_path)

from rice_ml.processing.preprocessing import (build_preprocessor_perceptron)
from rice_ml.supervised_learning.multilayer_perceptron import (train_mlp,
                                                               evaluate_model,
                                                               plot_precision_recall)
from rice_ml.supervised_learning.perceptron import (train_perceptron)


#### Basic Perceptron tests

# test if the perceptron converges on separable data
def test_linearly_separable_data():
    X = np.array([[2, 2], [3, 3], [-2, -2], [-3, -3]])
    y = np.array([1, 1, 0, 0])

    model = train_perceptron(X, y, max_iter=1000)
    metrics = evaluate_model(model, X, y)

    assert metrics["accuracy"] == 1.0

test_linearly_separable_data()


# test if the model understands what to do given a failure
def test_xor_limitation():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])

    model = train_perceptron(X, y, max_iter=1000)
    metrics = evaluate_model(model, X, y)

    assert metrics["accuracy"] < 1.0

test_xor_limitation()


# check that the training actually updates weights of variables
def test_weight_update_occurs():
    X = np.random.randn(20, 2)
    y = (X[:, 0] > 0).astype(int)

    model = train_perceptron(X, y, max_iter=50)

    assert np.any(model.coef_ != 0)

test_weight_update_occurs()



#### Multilayer Perceptron

# test the training model for mlp
def test_train_mlp():
    X, y, _ = load_and_prepare_data("adult.csv")
    preprocessor = build_preprocessor_perceptron(X)

    model, history, X_test, y_test = train_mlp(
        X, y, preprocessor, max_iter=50
    )

    assert isinstance(model, MLPClassifier)
    assert len(history) > 1
    assert model.loss_ > 0


# test model output for mlp
def test_evaluate_model():
    X, y, _ = load_and_prepare_data("adult.csv")
    preprocessor = build_preprocessor_perceptron(X)

    model, _, X_test, y_test = train_mlp(
        X, y, preprocessor, max_iter=50
    )

    metrics = evaluate_model(model, X_test, y_test)

    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1


# test precision-recall visualization
def test_plot_precision_recall():
    X, y, _ = load_and_prepare_data("adult.csv")
    preprocessor = build_preprocessor_perceptron(X)

    model, _, X_test, y_test = train_mlp(
        X, y, preprocessor, max_iter=20
    )

    try:
        plot_precision_recall(model, X_test, y_test)
    except Exception as e:
        pytest.fail(f"Precision-Recall plot failed: {e}")
