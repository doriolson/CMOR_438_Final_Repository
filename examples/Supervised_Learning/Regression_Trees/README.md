# Regression Trees

This directory contains example code and notes for the Regression Trees algorithm in supervised learning.

## Algorithm

Regression trees are the more complicated type of decision tree that are  used to make predictions. These trees create a flow chart (the tree) of various variables and outputs within the data until it thinks it can determine the correct answer of the binary income variable. After the algorithm follows an observation through the tree, it makes a prediction on the observation's income status based on the responses, assuming that those who have the same answers for all of the questions above will likely have the same income status. 

This algorithm assumes that observations will cluster themselves into similar groups, and that a group with the same responses for variables other than income will likely also have the same response for the income variable.

Information about regression trees from: https://www.youtube.com/watch?v=_wZ1Lo7bhGg


## Data

For this algorithm, we are using the adult income dataset. The data is loaded and processed in a standard manner, and a binary income variable is produced in order for the algorithm to be able to predict its values (although the algorithm predicts inclusive of 0 and 1, not just binary values). The remaining variables in the dataset may be used as part of the tree, but do not require additional preprocessing, as the model will interpret their results based on the characteristics it observes. Two variables of interest that the model "starts" its tree with (or are important in the tree organization) are age and number of hours worked. 