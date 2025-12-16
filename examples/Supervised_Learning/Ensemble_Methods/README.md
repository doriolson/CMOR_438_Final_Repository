## Ensemble Methods

This directory contains example code and notes for **Ensemble Methods**, a powerful class of machine learning algorithms that combine multiple individual models to produce more robust and accurate predictions. Ensemble methods work by aggregating the outputs of several base models, reducing the impact of errors and variability from any single model.

---

## Algorithm

Ensemble methods leverage the principle that multiple weak or diverse models, when combined, can outperform any individual model. There are two primary strategies for building ensembles:

- **Bagging (Bootstrap Aggregating)**: Multiple models (often decision trees) are trained in parallel on different bootstrap samples of the training data. Their predictions are then combined by majority vote (for classification) or averaging (for regression). Random Forests is a well-known example of a bagging ensemble.

- **Boosting**: Models are trained sequentially, with each new model focusing on correcting the errors of the previous ones. Observations that were misclassified or predicted poorly are given higher weight in subsequent iterations. Boosting methods include AdaBoost and Gradient Boosting. While highly effective, boosting can be sensitive to outliers because the model emphasizes correcting hard-to-predict points.

**Key Advantages**
- Reduces variance and increases prediction stability  
- Often achieves higher predictive accuracy than single models  
- Can combine diverse model types for more flexible learning  

**Limitations**
- Can be computationally expensive due to training multiple models  
- Less interpretable than a single model  
- Boosting can be sensitive to noisy data or outliers  

Ensemble methods are particularly useful when individual models have complementary strengths and weaknesses. By combining multiple learners, ensembles produce more consistent and reliable predictions than any single model on its own.

---

## Data

This project uses the **Adult Income dataset** to predict whether an individual earns more than \$50K per year. The dataset is first cleaned by replacing `"?"` entries with `NaN` and dropping all rows containing missing values to ensure complete data.  

Categorical features are converted to numeric values using **one-hot encoding**, and all features are standardized to ensure fair weighting during training. The target variable `income` is encoded as binary (`<=50K` → 0, `>50K` → 1).  

The prepared dataset is then split into **training and testing sets** with an 80/20 split and stratification to preserve class balance. This processed data is used to train and evaluate ensemble models like Random Forests and Boosting algorithms.
