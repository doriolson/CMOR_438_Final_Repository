## K Nearest Neighbors

This directory contains example code and notes for the **K Nearest Neighbors (KNN)** algorithm used in supervised learning. KNN is a distance-based approach that makes predictions by comparing new observations to similar examples in the training data rather than learning an explicit model during training.

---

## Algorithm

K Nearest Neighbors is a **non-parametric, instance-based** learning algorithm. Unlike models that estimate parameters during training, KNN stores all training data and performs computation only when a prediction is requested. For a new data point, the algorithm measures its distance to every point in the training set, selects the *k* closest neighbors, and uses their labels to determine the final prediction.

KNN supports both classification and regression tasks. In classification problems, the predicted class is determined by majority vote among the neighbors, while in regression settings, predictions are typically formed by averaging the neighbors’ target values. Because the method relies entirely on distance calculations, the choice of distance metric and feature scaling can have a significant impact on performance.

Several hyperparameters influence how KNN behaves:
- **k (number of neighbors)**, which controls the balance between sensitivity to local patterns and overall stability  
- **Distance metric**, such as Euclidean or Manhattan distance  
- **Neighbor weighting**, where closer points may be given more influence than farther ones  

### Pros
- Simple and intuitive algorithm with no explicit training phase  
- Flexible and capable of modeling non-linear decision boundaries  
- Makes minimal assumptions about the underlying data  

### Cons
- Computationally expensive at prediction time for large datasets  
- Highly sensitive to feature scaling and irrelevant variables  
- Performance can degrade in high-dimensional feature spaces  

---

## Data

This implementation uses the **Adult Income dataset**, which aims to predict whether an individual earns more than \$50K per year based on demographic and employment-related features.

The dataset is loaded from a local CSV file and cleaned by replacing `"?"` entries with missing values and removing any rows containing `NaN`. This ensures that all observations used by the model are complete, which is especially important for distance-based algorithms like KNN.

The target variable **`income`** is converted into a binary label, where incomes less than or equal to \$50K are mapped to 0 and incomes greater than \$50K are mapped to 1. Categorical variables such as workclass, education, occupation, marital status, and relationship are transformed using **one-hot encoding** so that all features are numeric and compatible with distance calculations.

Finally, the data is split into training and testing sets using an **80/20 split** with stratification to preserve class balance. The resulting datasets are then used to train and evaluate the KNN model.
