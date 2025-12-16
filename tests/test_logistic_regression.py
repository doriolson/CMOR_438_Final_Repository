
# Tests to run on the logistic regression / supervised learning model

# Testing for the data can be found in test_preprocessing.py

# Testing the model:

# Create Fake Boundary cases
def generate_boundary_cases(df):
    numeric_cols = df.select_dtypes(exclude="object").columns

    min_case = df[numeric_cols].min()
    max_case = df[numeric_cols].max()
    mean_case = df[numeric_cols].mean()

    base_case = df.mode().iloc[0]

    cases = []

    cases.append(base_case.copy())
    cases.append(base_case.copy())
    cases.append(base_case.copy())

    cases[0][numeric_cols] = min_case
    cases[1][numeric_cols] = max_case
    cases[2][numeric_cols] = mean_case

    boundary_df = pd.DataFrame(cases)
    boundary_df["case_type"] = ["MIN", "MAX", "MEAN"]

    return boundary_df


# Use the fake boundary cases to test the model
def stress_test_model(model, boundary_df):
    X_boundary = boundary_df.drop(["income", "case_type"], axis=1, errors="ignore")

    probs = model.predict_proba(X_boundary)[:, 1]
    preds = model.predict(X_boundary)

    results = boundary_df.copy()
    results["predicted_income"] = preds
    results["probability_>50K"] = probs

    print("\n=== Boundary Case Predictions ===")
    print(results[["case_type", "predicted_income", "probability_>50K"]])

    return results


# Probability sensitivity analysis
def feature_sensitivity(model, base_row, feature, values):
    rows = []

    for val in values:
        temp = base_row.copy()
        temp[feature] = val
        rows.append(temp)

    df_test = pd.DataFrame(rows)
    probs = model.predict_proba(df_test)[:, 1]

    plt.figure(figsize=(6, 4))
    plt.plot(values, probs, marker="o")
    plt.xlabel(feature)
    plt.ylabel("P(Income > 50K)")
    plt.title(f"Sensitivity Analysis: {feature}")
    plt.grid()
    plt.show()


# Out of Decision Test
def ood_test(model, df):
    ood_samples = df.sample(5, random_state=42).copy()

    ood_samples["age"] = 90
    ood_samples["hours-per-week"] = 1
    ood_samples["capital-gain"] = 999999

    probs = model.predict_proba(ood_samples.drop("income", axis=1))[:, 1]

    print("\n=== Out-of-Distribution Test ===")
    print(pd.DataFrame({
        "age": ood_samples["age"],
        "hours": ood_samples["hours-per-week"],
        "capital-gain": ood_samples["capital-gain"],
        "P(>50K)": probs
    }))


# Test the Decision Threshold 
def threshold_analysis(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 9)

    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_test, preds)
        results.append((t, acc))

    df_results = pd.DataFrame(results, columns=["Threshold", "Accuracy"])

    plt.figure(figsize=(6, 4))
    plt.plot(df_results["Threshold"], df_results["Accuracy"], marker="o")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Accuracy")
    plt.title("Threshold vs Accuracy")
    plt.grid()
    plt.show()

    return df_results


###########################
# Call all of the tests

# import the data and the model
import pandas as pd
df = pd.read_csv("adult.csv")

# create model
from rice_ml.processing.preprocessing import (preprocess_data)
from rice_ml.supervised_learning.logistic_regression import (train_model)
X, y, preprocessor = preprocess_data(df)

model, X_train, X_test, y_train, y_test = train_model(
    X, y, preprocessor
)


# Boundary case tests
boundary_df = generate_boundary_cases(df)
stress_test_model(model, boundary_df)

# Sensitivity analysis
base_row = df.drop("income", axis=1).iloc[0]
feature_sensitivity(
    model,
    base_row,
    feature="hours-per-week",
    values=np.arange(1, 80, 5)
)

# OOD behavior
ood_test(model, df)

# Threshold behavior
threshold_analysis(model, X_test, y_test)
