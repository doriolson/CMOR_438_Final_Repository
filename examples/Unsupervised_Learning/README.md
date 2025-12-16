## Unsupervised Learning Models

Unsupervised learning refers to machine learning tasks where data does not come with explicit labels or target values. The goal is to uncover structure in the data, such as clusters, lower-dimensional representations, or communities.

This directory contains examples of common unsupervised learning algorithms, including clustering methods, dimensionality reduction techniques, and community detection algorithms.

The algorithms included in this unsupervised machine learning repository are:
 - Community Detection
 - DBSCAN
 - K Means Clustering
 - PCA

These models use the obesity dataset described in the data folder of this repository. The data includes 2,111 observations of 17 variables, which are listed below: 

- `Age`, `Height`, `Weight`: Basic demographic and physical measurements.  
- `FCVC`: Frequency of vegetable consumption.  
- `NCP`: Number of main meals per day.  
- `CH2O`: Daily water consumption in liters.  
- `FAF`: Physical activity frequency.  
- `TUE`: Time using electronic devices.  
- `CALC`, `FAVC`, `SCC`, `SMOKE`: Lifestyle habits and behaviors.  
- `family_history_with_overweight`: Indicates presence of overweight family members.  
- `CAEC`: Consumption of high-calorie food.  
- `MTRANS`: Mode of transportation.  
- `NObeyesdad`: Obesity classification (used only for evaluation).  