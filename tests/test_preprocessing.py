import numpy as np
import pytest

from rice_ml import (
    standardize,
    minmax_scale,
    maxabs_scale,
    l2_normalize_rows,
    train_test_split,
    train_val_test_split,
)


# ---------------------- Scaling & Normalization ----------------------

def test_standardize_basic_and_params():
    X = np.array([[1., 2.], [3., 2.], [5., 2.]])
    Z, params = standardize(X, return_params=True)
    assert Z.shape == X.shape
    # Column 1 has variance, column 2 is constant
    assert np.allclose(Z[:, 1], 0.0)
    assert params["scale"][1] == 1.0  # zero variance handled
    # Centered means ~0
    assert np.allclose(Z.mean(axis=0), 0.0)


def test_standardize_no_std_or_mean():
    X = np.array([[1., 2.], [3., 4.]])
    Z = standardize(X, with_mean=False, with_std=False)
    assert np.allclose(Z, X)
    Z = standardize(X, with_mean=True, with_std=False)
    assert not np.allclose(Z, X)
    assert np.allclose(Z.mean(axis=0), 0.0)


def test_minmax_scale_range_and_params():
    X = np.array([[0., 10.], [5., 10.], [10., 10.]])
    X2, params = minmax_scale(X, feature_range=(2, 3), return_params=True)
    assert X2.shape == X.shape
    # First feature maps 0->2, 10->3
    assert np.allclose(X2[:, 0], [2.0, 2.5, 3.0])
    # Second feature zero-range -> mapped to lower bound
    assert np.allclose(X2[:, 1], 2.0)
    assert params["feature_range"] == (2.0, 3.0)
    assert params["scale"][1] == 1.0


def test_maxabs_scale_basic():
    X = np.array([[-2., 0.], [1., 0.], [2., 0.]])
    X2, params = maxabs_scale(X, return_params=True)
    assert np.allclose(X2[:, 0], [-1.0, 0.5, 1.0])
    assert np.allclose(X2[:, 1], [0.0, 0.0, 0.0])
    assert params["scale"][1] == 1.0


def test_l2_normalize_rows_behavior():
    X = np.array([[3., 4.], [0., 0.]])
    Xn = l2_normalize_rows(X)
    assert np.isclose(np.linalg.norm(Xn[0]), 1.0)
    assert np.allclose(Xn[1], [0.0, 0.0])
    with pytest.raises(ValueError):
        l2_normalize_rows(X, eps=0.0)


def test_scalers_input_validation():
    with pytest.raises(ValueError):
        standardize(np.array([1., 2., 3.]))  # not 2D
    with pytest.raises(TypeError):
        standardize([["a", "b"], ["c", "d"]])
    with pytest.raises(ValueError):
        minmax_scale(np.empty((0, 2)))
    with pytest.raises(ValueError):
        minmax_scale(np.ones((2, 2)), feature_range=(1, 1))
    with pytest.raises(ValueError):
        l2_normalize_rows(np.ones((2, 2)), eps=-1.0)


# ---------------------- Splitting ----------------------

def test_train_test_split_shapes_and_determinism():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, test_size=0.3, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.3, random_state=42)
    assert X_tr1.shape == (35, 2)
    assert X_te1.shape == (15, 2)
    assert np.array_equal(y_tr1, y_tr2)
    assert np.array_equal(X_te1, X_te2)
    # Without shuffle=False, we still should get consistent sizes
    X_tr3, X_te3 = train_test_split(X, test_size=0.2, shuffle=False)
    assert X_tr3.shape == (40, 2) and X_te3.shape == (10, 2)


def test_train_test_split_stratify():
    X = np.arange(60).reshape(30, 2)
    y = np.array([0, 1, 2] * 10)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    # Each class appears in both splits
    assert set(np.unique(y_tr)) == set(np.unique(y_te))
    # Proportions roughly preserved
    from collections import Counter
    c_full = Counter(y)
    c_tr = Counter(y_tr)
    c_te = Counter(y_te)
    for k in c_full:
        # Within 1 sample of expected proportional split
        expected_te = round(0.3 * c_full[k])
        assert abs(c_te[k] - expected_te) <= 1


def test_train_val_test_split_shapes_and_stratify():
    X = np.arange(90).reshape(45, 2)
    y = np.array([0, 1, 2] * 15)
    parts = train_val_test_split(X, y, val_size=0.2, test_size=0.2, stratify=y, random_state=123)
    X_tr, X_va, X_te, y_tr, y_va, y_te = parts
    assert X_tr.shape == (27, 2)
    assert X_va.shape == (9, 2)
    assert X_te.shape == (9, 2)
    assert set(np.unique(y_tr)) == set(np.unique(y_va)) == set(np.unique(y_te))


def test_split_input_validation():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    with pytest.raises(ValueError):
        train_test_split(np.arange(10), y)  # not 2D X
    with pytest.raises(ValueError):
        train_test_split(X, y[:-1])  # shape mismatch
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)
    with pytest.raises(TypeError):
        train_test_split(X, y, random_state="seed")
    with pytest.raises(ValueError):
        train_val_test_split(X, y, val_size=0.6, test_size=0.5)
    with pytest.raises(ValueError):
        train_val_test_split(X, y, val_size=-0.1, test_size=0.2)


def test_train_val_test_split_without_y():
    X = np.arange(30).reshape(15, 2)
    X_tr, X_va, X_te = train_val_test_split(X, val_size=0.2, test_size=0.2, random_state=7)
    assert X_tr.shape == (9, 2)
    assert X_va.shape == (3, 2)
    assert X_te.shape == (3, 2)
    




########################################
# testing the supervised ML data
### Testing the data
import pandas as pd

# set path to find dataset
repo_root = Path("/Users/doriolson/Desktop/repos/CMOR_438_Final_Repository")
data_path = Path("../../../Data/adult.csv")

df = pd.read_csv(data_path)

# Analysis of Dataset Boundaries
def dataset_boundaries(df):
    summary = df.describe(include="all").transpose()

    print("\n=== Dataset Boundaries ===")
    print(summary[["min", "max"]])

    return summary

dataset_boundaries(df)

# Check for rare or invalid values
def check_invalid_values(df):
    issues = {}

    issues["negative_age"] = df[df["age"] < 0].shape[0]
    issues["zero_hours"] = df[df["hours-per-week"] == 0].shape[0]
    issues["extreme_hours"] = df[df["hours-per-week"] > 100].shape[0]
    issues["negative_capital_gain"] = df[df["capital-gain"] < 0].shape[0]
    issues["negative_capital_loss"] = df[df["capital-loss"] < 0].shape[0]

    print("\n=== Invalid / Edge Values ===")
    for k, v in issues.items():
        print(f"{k}: {v}")

    return issues

check_invalid_values(df)


#### perceptron preprocessing
# load data
from rice_ml.processing.preprocessing import (load_and_prepare_data,
                                              build_preprocessor_perceptron)
from rice_ml.supervised_learning.multilayer_perceptron import train_mlp

def test_load_and_prepare_data():
    X, y, feature_names = load_and_prepare_data(data_path)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert y.nunique() == 2
    assert X.isnull().sum().sum() == 0



# test preprocessing function
def test_build_preprocessor_perceptron():
    X, _, _ = load_and_prepare_data("adult.csv")
    preprocessor = build_preprocessor_perceptron(X)

    assert preprocessor is not None
    assert hasattr(preprocessor, "fit")
    assert hasattr(preprocessor, "transform")


# test data boundaries to ensure the model can run with a small dataset
def test_small_dataset_behavior():
    X, y, _ = load_and_prepare_data("adult.csv")

    X_small = X.iloc[:100]
    y_small = y.iloc[:100]

    preprocessor = build_preprocessor_perceptron(X_small)

    model, _, _, _ = train_mlp(
        X_small, y_small, preprocessor, max_iter=20
    )

    assert model is not None
    

# test extreme input values in the data
def test_extreme_values():
    X, y, _ = load_and_prepare_data("adult.csv")
    preprocessor = build_preprocessor_perceptron(X)

    model, _, X_test, y_test = train_mlp(
        X, y, preprocessor, max_iter=20
    )

    X_test_extreme = X_test.copy()
    X_test_extreme["age"] = 90
    X_test_extreme["hours-per-week"] = 1
    X_test_extreme["capital-gain"] = 1_000_000

    preds = model.predict(X_test_extreme)
    assert len(preds) == len(X_test_extreme)


# run the tests
