# K Means Clustering

This directory contains example code and notes for the K Means Clustering algorithm
in unsupervised learning.

## Algorithm


K means clustering is an algorithm designed to organize the data into k groups (or clusters) with the assumption that these groups will uncover hidden patterns within the data. These groups are created based on the eucledian distance each observation is from the "cluster center." Ideally, at the end the end of this process each data point will be more similar to the other points in its cluster than points in any other cluster. 

The algorithm follows the following steps:
1. Initial Guess - start with randomly determined cluster centers
2. Assignment - assign each data point to the nearest cluster center (measured in eucledian disatnce)
3. Update - recalculate the cluster centers (centroids) by averaging the distances of points within each cluster
4. Repeat - until a set maximum number of iterations (50), or the centroids stop changing (no longer need to update)

The most important parameter for a k-means clustering algorithm is the value of k to use. In this model, we ran a elbow chart and a silhouette analysis, which determined that the ideal number of clusters for this data is 4. 


Source of information on k-means clustering: http://geeksforgeeks.org/machine-learning/k-means-clustering-introduction/. 

## Data


For this algorithm, we will be using the obesity dataset. Since this is an unsupervised learning model, we will remove the variable NObeyesdad, which is an indicator of whether an observation is obese or not. The rest of the variables will be included in the k means clustering, and will be sorted into numerical and categorical variables. The numerical and categorical variables are preprocessed before the training begins, using functions that can be found in the src folder (as are all of the functions used in the machine learning).