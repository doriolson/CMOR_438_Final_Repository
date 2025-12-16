
# Read me: data testing

This folder contains a series of tests we conducted on our data to test our models and functions before running them through the full datasets. This is a non comprehensive list of tests, but intends to predict common or potential errors in our code and mitigate any problems before they would appear in our "real" code in the example folder. 

We ran a combination of test on the datasets and tests on aspects of the machine learning algorithms. Most of the data-based tests can be found in the test_preprocessing.py file, and include tests of :
 - Small datasets
 - Outlier data
 - Missing values
 - Invalid values
 - Data loading

 The tests we ran on the machine learning algorithms themselves evaluated specific functions within our models on a smaller scale. These tests intended to evaluate:
 - Training functions
 - Evaluation functions
 - Visualization functions


The tests in this folder are intended to be used after the code and functions are created in the src folder and before the algorithms are implemented in the examples folder.