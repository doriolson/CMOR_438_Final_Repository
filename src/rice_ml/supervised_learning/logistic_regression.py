# This is the src code for logistic regression


#first, load in libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


##################################
# here I am loading in the dataset
# this will be moved eventually
# the file path will be "adult.csv"
# the variable to focus on will be "incomebinary"
def load_dataset(file_path):
    df = pd.read_csv(file_path)

    # Clean whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Encode target variable
    df["incomebinary"] = df["income"].map({"<=50K": 0, ">50K": 1})

    return df


######################################################################
# Here are the functions that will go into the model training function


# split data - technically pre-processing, move
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and test sets.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

# preprocess - move to pre-processing
def preprocess_data(X_train, X_test):
    """
    Encode categorical features and scale numerical features.
    """
    # Detect column types
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    # Create preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor

# train the model
def train_logistic_regression(X_train, y_train):
    """
    Train logistic regression model.
    """
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# make predictions
def predict(model, X):
    """
    Predict class labels and probabilities.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob

# evaluate
def evaluate_model(y_true, y_pred):
    """
    Print evaluation metrics.
    """
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))




###########################################
# This is the function to train the real dataset
def train_pipeline(file_path, target_column):
    """
    Complete training pipeline using real dataset.
    """

    # Load data
    X = load_dataset(file_path)
    y = target_column

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Preprocess
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # Train
    model = train_logistic_regression(X_train_scaled, y_train)

    # Evaluate
    y_pred, _ = predict(model, X_test_scaled)
    evaluate_model(y_test, y_pred)

    return model, scaler


########################################
# Here is a graph visualization of the logistic regression
def plot_probability_distribution(results):
    """
    Plot predicted probability distribution.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=results,
        x="probability",
        hue="prediction",
        bins=30,
        kde=True
    )
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Logistic Regression Probability Distribution")
    plt.show()

plot_probability_distribution(results)

