# Perceptron

This directory contains example code and notes for the Perceptron algorithm
in supervised learning.

## Algorithm


Perceptron is an algorithm that takes multiple inputs, gives them each certain weights, and then outputs a binary decision based on if the sum of the weighted inputs meets a certain threshhold. This algorithm is the basis for a number of other machine learning processes. Perceptron requires a vector of other variables to use as predictors, the weights that should be given to each of these other variables, (potentially) a bias term, and a threshold for the sum of the values and weights to reach to determine the binary outcome. 

The algorithm will be predicting the income of each person/observation using the remaining 14 variables, weighted by importance level. The model will run tests to determine how each variable should be weighted to help make the most accurate prediction, and then apply these weights given a threshold to the larger data set. Upon this, the models accuracy can be determined.  

Source on the algorithm: https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/. 


## Data

For this algorithm I am using the adult income database descrribed in the supervised learning README.md. The variable of interest is income, which will be converted into a binary variable for the purposes of the algorithm and the remaining variables will be divided as numerical and categorical to help determine their weighting. 