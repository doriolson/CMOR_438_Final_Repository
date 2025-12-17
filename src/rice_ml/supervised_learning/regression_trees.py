# Here is the code for a regression tree algorithm

# first, load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             accuracy_score)
from sklearn.tree import plot_tree

from matplotlib.colors import ListedColormap


# Import necessary preprocessing functions for the execution block
# NOTE: These must be imported into the module's namespace if the functions are called locally.
try:
    from rice_ml.processing.preprocessing import load_and_prepare_data, build_preprocessor
except ImportError:
    # If the import fails, the execution block will skip, but functions should remain callable.
    pass 


############### Functions ###################

# Here is the function to train the regression tree
def train_regression_tree(df):
    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Assumes build_preprocessor is imported or accessible
    preprocessor = build_preprocessor() 

    tree = DecisionTreeRegressor(
        max_depth=6,
        min_samples_leaf=50,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", tree)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


# Here is the function to evaluate the regression tree
def evaluate_regression_tree(model, X_test, y_test):
    y_pred_cont = model.predict(X_test)
    y_pred_class = (y_pred_cont >= 0.5).astype(int)

    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_cont))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_cont))
    print("R² Score:", r2_score(y_test, y_pred_cont))
    print("Classification Accuracy:", accuracy_score(y_test, y_pred_class))


############# Visualizations ################
# visualize the top levels of the tree structure
def plot_regression_tree(model, max_depth=3):
    tree = model.named_steps["regressor"]
    feature_names = model.named_steps["preprocess"].get_feature_names_out()

    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        filled=True,
        max_depth=max_depth,
        rounded=True
    )
    plt.title("Regression Tree (Top Levels)")
    plt.show()


# feature importance plot
def plot_feature_importance(model, top_n=15):
    tree = model.named_steps["regressor"]
    importances = tree.feature_importances_
    feature_names = model.named_steps["preprocess"].get_feature_names_out()

    idx = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(8, 5))
    plt.barh(range(top_n), importances[idx])
    plt.yticks(range(top_n), feature_names[idx])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances (Regression Tree)")
    plt.show()


# true vs predicted scatter plot
def plot_true_vs_predicted(model, X_test, y_test):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("True Income")
    plt.ylabel("Predicted Income")
    plt.title("True vs Predicted (Regression Tree)")
    plt.show()


# this visual is how the model should supposedly be splitting into decisions
def plot_tree_splits_2d(
    pipeline,
    df,
    feature_x="age",
    feature_y="hours-per-week",
    grid_size=300
):
    """
    Correct visualization of decision tree splits in 2D
    using the trained preprocessing + tree pipeline.
    """
    # Separate features and target
    X = df.drop(columns=["income"])
    y = df["income"]

    # Create grid range for the background decision regions
    x_min, x_max = X[feature_x].min(), X[feature_x].max()
    y_min, y_max = X[feature_y].min(), X[feature_y].max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    # Build a baseline row (mean / mode) for the features NOT being plotted
    baseline = {}
    for col in X.columns:
        if col == feature_x:
            baseline[col] = xx.ravel()
        elif col == feature_y:
            baseline[col] = yy.ravel()
        else:
            if X[col].dtype == "object":
                baseline[col] = X[col].mode()[0]
            else:
                baseline[col] = X[col].mean()

    grid_df = pd.DataFrame(baseline)

    # Predict using the FULL PIPELINE (including preprocessing)
    Z = pipeline.predict(grid_df)
    Z = Z.reshape(xx.shape)

    # --- THE PLOTTING SECTION (This was missing/incomplete) ---
    plt.figure(figsize=(9, 7))
    
    # Define custom colors: Light Blue for <=50K, Light Red for >50K
    cmap_bg = ListedColormap(["#cce5ff", "#ffcccc"])
    
    # Draw the background decision regions
    # Regression trees output continuous values; we threshold at 0.5 for class regions
    plt.contourf(xx, yy, Z >= 0.5, alpha=0.6, cmap=cmap_bg)

    # Draw the actual data points from the original dataframe
    plt.scatter(
        X[feature_x],
        X[feature_y],
        c=y,
        cmap="coolwarm",
        edgecolor="k",
        alpha=0.6,
        s=20
    )

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"Decision Tree Split Regions: {feature_x} vs {feature_y}")
    plt.colorbar(label="Actual Income Class (0: <=50K, 1: >50K)")
    
    # Critical: This command forces the window to render in Jupyter
    plt.show()
