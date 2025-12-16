# PCA

This directory contains example code and notes for the PCA algorithm
in unsupervised learning.

## Algorithm

Principal Component Analysis (or PCA) is a dimensionality reduction tool that maintains the basic premise of the dataset but simplifies the noise and complexity of variables so that audience can better understand the data. PCA uses linear algebra (eigenvectors) to find the principal components of the data, or the variables to keep when simplifying the dataset. 

The steps involved in Principal Component Analysis are as follows:
1. Standardize the data into eigenvectors using z-scores 
2. Calculate a covariate matrix to see the relations between each of the variables
3. Identify the principal components, ranked by importance using the eigenvalues and matrix
4. Transform the data onto the principal components

Once these steps are complete, you should have a new, simplified dataset that only involves a few variables and is thus easier to interpret and implement. 

Our analysis created two principal components:
 - PC1 measures lifestyle and body mass: weight, height, frequency of physical activity, and dietary behaviors.
 - PC2 measures consumption habitsy: hydration habits, transportation, and type spent sedentary.

 Source for information on the PCA algorithm: https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/. 

## Data


This algorithm uses the obesity dataset. There was very little preprocssing involved, though all of the variables were divided into numerical and categorical data. Notably, the obesity variable was excluded for the data training and testing, but used to provide insight to the data visualization 