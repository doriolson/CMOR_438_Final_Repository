# Multilayer Perceptron

This directory contains example code and notes for the Multilayer Perceptron algorithm in supervised learning.

## Algorithm

A multilayer perceptron is a series of connected perceptron models (called neurons), in which each of the perceptrons use the inputs and outputs of the models around them (the outputs of one are the inputs of the next "layer," and so forth). This is also known as a neural network. 

The MLP functions very similarly to the perceptron, but on a larger scale. Like a perceptron, it takes inputs and assigns them various weights to try to predict a binary outcome. A multilayer perceptron also implements a practice called back propogation. This means that if one perceptron  algorithm realizes its initial guess (or output) was wrong, it will then go back through all of the perceptron levels it needed to pass to reach that point, and redo those perceptrons with the udpated estimate. 

My source of information about MLPs is here: https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron. 

## Data


As the MLP is a somewhat more complicated version of perceptron, they will be using the same data on adult income. The target variable to test will be a binary measure of adult income, and the other variables in the dataset will be weighted according to the model's specifications. More information about the dataset and use of these variables can be found in the README.md for perceptron and the general supervised learning folder.