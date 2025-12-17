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

# Assuming these functions are available via import from the package runner script
# (e.g., from rice_ml.processing.preprocessing)
# We need to import them here since they are used by the run block and other functions below.
# Note: These lines assume load_and_prepare_data and build_preprocessor_perceptron are defined 
# elsewhere (i.e., in preprocessing.py) and available globally via the package import structure.
# If they are NOT in preprocessing.py, you need to add the imports here.

######################## functions to be used #####################
# NOTE: The load_and_prepare_data and build_preprocessor_perceptron functions
# are typically defined in rice_ml.processing.preprocessing.
# We REMOVE them here to avoid duplication/confusion with the notebook's import.

# function to train the perceptron algorithm
def train_perceptron(df):
    X = df.drop(columns=["income"])
    y = df["income"]

    # This requires build_preprocessor_perceptron to be defined or imported. 
    # Since it was defined above in your old file, we assume it's now imported or globally accessible.
    # For safety, we rely on the notebook's import for build_preprocessor_perceptron.
    # If the function is not imported into this file's namespace, it will fail.
    # To fix this, we import it locally:
    # from rice_ml.processing.preprocessing import build_preprocessor_perceptron 
    
    # We will assume that the function you had defined locally is the required implementation:
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
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor_perceptron() # Use the locally defined one

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



############### Actually run it (Guarded and Corrected) ########################
if __name__ == "__main__":
    # FIX 1: Corrected relative path for execution from notebook's CWD
    DATA_FILE = "../../../Data/adult.csv" 
    
    # FIX 2: Assuming load_and_prepare_data is available via your package's __init__.py chain,
    # or you need to import it here if the __init__.py chain is broken.
    # For safety, we will locally import the necessary component for the run block:
    try:
        from rice_ml.processing.preprocessing import load_and_prepare_data
    except ImportError:
        print("Warning: load_and_prepare_data not found via module import. Skipping run code block.")
        exit()
    
    # upload the file from csv format
    # Removed: uploaded = files.upload() (Google Colab code)

    # actually run all the things
    df = load_and_prepare_data(DATA_FILE)
    model, X_test, y_test = train_perceptron(df)
    evaluate_model(model, X_test, y_test)
    
    # run visualizations
    plot_decision_boundary(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
