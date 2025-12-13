# Logistic Regression

This directory contains example code and notes for the Logistic Regression algorithm in supervised learning.

## Algorithm

_TODO: Describe the core idea of Logistic Regression, its objective, and key hyperparameters._

Logistic Regression is used to classify or predict the outcome of a binary variable. Rather than a straight line of a linear regression, it uses a sigmoid function (curved line) to more accurately capture the tendencies of the data. 

This algorithm relies on a few assumptions or parameters:
 - The dependent variable is binary and can only have two classifications
 - Each data point is independent from the other
 - The data does not have extreme outliers that may skew the sigmoid
 - The data set is large enough to accurately train a model

 Most significantly, logistic regression requires one binary variable for their measurement, so I am using the gender variable. This means that the model will predict the gender of the observation based on their answers to all of the other factors in the dataset. 


More information (and my source for this analysis) can be found here: https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/. 

## Data

For this model, I am using the dataset on adult incomes, which is described in the general supervised learning README.md file. As per the model, I will be focusing specifically on one binary variable. Given the scope of this dataset, I will use gender as the binary variable. The current gender variable is a string and not formatted for a logistic regression, so preprocessing will have to involve converting the gender variable into a binary measurement. 
