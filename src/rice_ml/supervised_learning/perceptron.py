# Perceptron Algorithm code

## load necessary python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.metrics import ( accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA


######################## functions to be used #####################

# function to load and clean data 
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Clean whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Encode target variable
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    return df


# function for preprocessing
def build_preprocessor_perceptron():
    numeric_features = [
        "age", "fnlwgt", "educational-num",
        "capital-gain", "capital-loss", "hours-per-week"
    ]

    categorical_features = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race",
        "gender", "native-country"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor


# function to train the perceptron algorithm
def train_perceptron(df):
    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor_perceptron()

    model = Perceptron(
        max_iter=1000,
        eta0=1.0,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


# function to evaluate data
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


############### Actually run it ########################

# upload the file from csv format
# this is designed for google notebook (w extra library)
uploaded = files.upload()

data = load_and_prepare_data("adult.csv")

# actually run all the things
df = load_and_prepare_data("adult.csv")
model, X_test, y_test = train_perceptron(df)
evaluate_model(model, X_test, y_test)


############### Visualization ####################
def plot_decision_boundary(model, X, y):
    # Transform features
    X_transformed = model.named_steps["preprocess"].transform(X)

    # Reduce to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_transformed.toarray())

    # Predict
    y_pred = model.named_steps["classifier"].predict(X_transformed)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="coolwarm", alpha=0.5)
    plt.title("Perceptron Decision Boundary (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Predicted Income Class")
    plt.show()

plot_decision_boundary(model, X_test, y_test)


# confusion matrix heat map
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Perceptron Confusion Matrix")
    plt.show()
