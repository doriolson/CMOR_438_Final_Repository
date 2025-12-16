## DBSCAN

This directory contains example code and notes for **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**, an unsupervised learning algorithm used for clustering data based on density. DBSCAN identifies clusters as regions of high point density and treats points in low-density regions as outliers or noise. Unlike methods such as k-means, DBSCAN does not require specifying the number of clusters in advance and can detect clusters of arbitrary shapes.

---

## Algorithm

DBSCAN groups data points based on two key parameters: **eps** (the maximum distance between two points to consider them neighbors) and **min_samples** (the minimum number of points required to form a dense region). The algorithm proceeds as follows:

1. Identify **core points**: points that have at least `min_samples` neighbors within `eps` distance.  
2. Starting from an unvisited core point, form a cluster by recursively including all core points reachable within `eps` distance.  
3. Include **border points**, which are not core points themselves but fall within `eps` of a core point.  
4. Repeat the process until all core points are assigned to clusters. Any remaining points that are not reachable from a core point are considered **noise**.

**Key Hyperparameters**
- **eps**: Distance threshold for neighborhood inclusion  
- **min_samples**: Minimum number of neighbors to form a dense region  
- **metric**: Distance metric used (commonly Euclidean)  

**Pros**
- Can find clusters of arbitrary shape, not limited to spherical clusters  
- Automatically detects outliers  
- Does not require specifying the number of clusters beforehand  

**Cons**
- Sensitive to the choice of `eps` and `min_samples`  
- Performance can degrade on high-dimensional data  
- Not ideal for datasets with varying density clusters  

DBSCAN is particularly useful when the data contains irregularly shaped clusters or when noise detection is important.

---

## Data

For this project, we use an **Obesity dataset** to explore clustering patterns across different health-related features.  

Categorical features are converted into numeric format using **one-hot encoding**, and all numeric features are **standardized** using `StandardScaler` to ensure that distance calculations are meaningful. The resulting scaled feature matrix is then used as input to the DBSCAN algorithm.

After fitting DBSCAN, each data point is assigned a cluster label, with `-1` indicating noise. The algorithm produces clusters of varying sizes and identifies points that do not belong to any cluster, allowing for an analysis of both dense regions and outliers in the dataset.
