## Linear Regression

This directory contains example code and notes for **Linear Regression**, one of the simplest and most widely used supervised learning algorithms. Linear regression models the relationship between one or more input features and a continuous target variable by fitting a linear equation to the observed data. The model assumes that the target can be approximated as a linear combination of the input features plus an error term.

---

## Algorithm

Linear Regression estimates a set of coefficients for the input features that minimize the difference between predicted and observed target values. Mathematically, the model can be expressed as:

\[
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} + \epsilon_i
\]

where \(y_i\) is the target, \(x_{ij}\) are feature values, \(\beta_j\) are the coefficients, and \(\epsilon_i\) represents the error term. The algorithm aims to minimize the **mean squared error** (MSE) between predicted and actual target values. This is typically solved using the ordinary least squares method or iterative optimization techniques such as **gradient descent**.

Linear regression assumes:
- Linearity: the relationship between features and target is linear in the coefficients  
- Independence of errors: residuals are uncorrelated  
- Homoscedasticity: constant variance of errors across predictions  
- Normality of errors (for inference and confidence intervals)

**Pros**
- Simple and interpretable model, easy to visualize  
- Fast to train and compute predictions  
- Provides a clear understanding of feature impact via coefficients  

**Cons**
- Assumes linear relationships, which may not hold in many datasets  
- Sensitive to outliers and correlated features  
- Cannot naturally model complex or non-linear patterns  

Linear regression is best suited for predicting continuous variables and serves as a strong baseline for many supervised learning problems.

---

## Data

For this project, we use the **Adult Income dataset** to predict whether an individual earns more than \$50K per year. The dataset is first cleaned by replacing `"?"` entries with missing values and dropping incomplete rows.  

Categorical features are converted into numeric format using **one-hot encoding**, and the target variable `income` is encoded as binary (`<=50K` → 0, `>50K` → 1). The cleaned dataset is then split into **training and testing sets** using an 80/20 split with stratification to preserve class balance.  

Finally, all features are **standardized** to ensure that each variable contributes equally to the linear model, which is particularly important for models that compute coefficients based on gradient-based optimization.
