# This is the src code for logistic regression


# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


######### preprocessing #########

# load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# preprocess
def preprocess_data(df):
    X = df.drop("income", axis=1)
    y = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return X, y, preprocessor


############# Training data ###########
def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test



### evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return y_pred, y_prob


########## visualizations
# confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ROC curve
def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# feature importance aka top coefficients
def plot_top_coefficients(model, preprocessor, top_n=15):
    feature_names = (
        preprocessor
        .transformers_[0][2].tolist() +
        list(
            preprocessor
            .transformers_[1][1]
            .named_steps["onehot"]
            .get_feature_names_out()
        )
    )

    coefficients = model.named_steps["classifier"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    }).sort_values(by="coefficient", key=abs, ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x="coefficient",
        y="feature",
        data=coef_df,
        palette="coolwarm"
    )
    plt.title("Top Logistic Regression Coefficients")
    plt.show()


######### run code

if __name__ == "__main__":
    
    # 1. Load Data
    # Path is now fixed to be relative to the notebook's CWD
    df = pd.read_csv("../../../Data/adult.csv")
    
    # 2. Preprocess
    # Assuming preprocess_data is available globally
    X, y, preprocessor = preprocess_data(df)

    # 3. Train Model
    # Assuming train_model is available globally
    model, X_train, X_test, y_train, y_test = train_model(
        X, y, preprocessor
    )
    
    # 4. Evaluate Model (THIS LINE MUST BE INSIDE THE GUARD)
    y_pred, y_prob = evaluate_model(model, X_test, y_test)

    # 5. Visualize (THESE LINES MUST ALSO BE INSIDE THE GUARD)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_top_coefficients(model, preprocessor)



