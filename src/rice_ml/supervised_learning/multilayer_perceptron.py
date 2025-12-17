# Here is the code for a multilayer perceptron algorithm

# load in the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve


######## Pre-processing (moved to preprocessing, same as perceptron)
# load in data function
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Remove leading/trailing spaces
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Encode target
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    return df

# preprocessing function
# preprocessor (same as other)
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



########## Training Function #########
def train_mlp(df):
    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor_perceptron()

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=50,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", mlp)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


######### evaluation function ############
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


######### visualizations ############

# PCA Projection of Learned Representation
def plot_pca_representation(model, X, y):
    X_transformed = model.named_steps["preprocess"].transform(X)

    # Convert sparse → dense (safe for PCA)
    X_dense = X_transformed.toarray()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_dense)

    y_pred = model.predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_pred,
        cmap="coolwarm",
        alpha=0.5
    )
    plt.title("MLP Decision Regions (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Predicted Income Class")
    plt.show()


# training loss curve
# shows convergence, and risk of overfitting

def plot_training_loss(model):
    mlp = model.named_steps["classifier"]
    plt.figure(figsize=(7, 5))
    plt.plot(mlp.loss_curve_)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.show()


# Precision-Recall Curve
# good for imbalanced data, which is true of adult dataset
def plot_precision_recall(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (MLP)")
    plt.show()


########## create the model and run the script ###########

if __name__ == "__main__":
    # FIX 1: Corrected relative path for execution from notebook's CWD
    DATA_FILE = "../../../Data/adult.csv" 
    
    # load in dataset
    df = load_and_prepare_data(DATA_FILE)

    # train the model
    model, X_test, y_test = train_mlp(df)

    # evaluate the model
    evaluate_model(model, X_test, y_test)

    # run the visualizations (FIX 2: All execution code is consolidated here)
    plot_pca_representation(model, X_test, y_test)
    plot_training_loss(model)
    plot_precision_recall(model, X_test, y_test)