## Decision Trees

This directory contains example code and notes for **Decision Tree algorithms**, a versatile supervised learning method used for both classification and regression. Decision trees model decisions using a tree-like structure, where each internal node represents a feature-based split, and each leaf node represents a prediction outcome. The algorithm is intuitive and interpretable, making it a popular choice for many practical problems.

---

## Algorithm

Decision Trees are **non-parametric, hierarchical** models that recursively split data to create branches and leaf nodes. The goal is to partition the dataset into subsets that are as homogeneous as possible with respect to the target variable.  

For **classification tasks**, splits are chosen to maximize **information gain**, which measures the reduction in entropy after a split:

\[
IG = H(\text{parent}) - \sum_i w_i H(\text{child}_i)
\]

where \(H\) is the entropy and \(w_i\) represents the proportion of samples in each child node.  

For **regression tasks**, splits are determined by **variance reduction**, which minimizes uncertainty in the target values across child nodes:

\[
VR = \text{Var(parent)} - \sum_i w_i \text{Var(child}_i)
\]

The tree grows by repeatedly splitting nodes until stopping criteria are met, such as reaching a **maximum depth**, a **minimum number of samples per node**, or negligible improvement in information gain or variance reduction. Predictions are made by traversing the tree from the root to a leaf and outputting the class label or average target value for that leaf.

**Key Hyperparameters**
- **max_depth**: Maximum depth of the tree to prevent overfitting  
- **min_samples_split**: Minimum number of samples required to split a node  
- **min_samples_leaf**: Minimum number of samples required to form a leaf node  
- **max_features**: Maximum number of features considered for splitting at each node  

### Pros
- Highly interpretable and easy to visualize  
- Handles both numerical and categorical data  
- Captures non-linear relationships without requiring feature scaling  

### Cons
- Prone to overfitting if not properly regularized  
- Small changes in data can lead to very different trees  
- Can be biased toward features with more levels  

---

## Data

This project uses the **Adult Income dataset** to predict whether an individual earns more than \$50K per year based on demographic and employment features.  

The dataset is first cleaned by replacing `"?"` entries with `NaN` and dropping all rows containing missing values, ensuring complete data for training and evaluation. The target variable `income` is encoded as a binary label (`<=50K` → 0, `>50K` → 1). Categorical features such as workclass, education, occupation, marital status, and relationship are converted to numeric format using **one-hot encoding**, resulting in a dataset fully compatible with the decision tree algorithm.  

Finally, the data is split into **training and testing sets** using an 80/20 split with stratification to preserve class balance. This prepared dataset is then used to train and evaluate the decision tree model.
