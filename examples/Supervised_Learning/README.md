## Supervised Learning Models

Supervised learning refers to machine learning tasks where each example comes with an associated label or target value. The goal is to learn a mapping from inputs (features) to outputs (labels) so that the model can make accurate predictions on new, unseen data.

This directory contains examples of common supervised learning algorithms for both regression and classification. These models include:
 - linear regression
 - logistic regression
 - decision trees
 - k-nearest neighbors
 - ensemble methods
 - perceptron
 - multilayer perceptron
 - regression trees


################

## Data

The supervised learning algorithms use the adult income dataset. This dataset contains 48,842 observations of 15 variables. More information can be found in the data folder. The variables in the data are: 

 - `age`: age of the person in years
 - `workclass`: what type of employment (private, government, self-employed, etc)
 - `fnlwgt`: final weight, a scaling factor to know how much weight that observation (or person's data) carries
 - `education`: what was your highest degree or level of education
 - `educational-num`: numerical representation of education in years
 - `marital-status`: marriage status (married, widowed, etc)
 - `occupation`: job or general career field
 - `relationship`: what role are you in your household (own-child, husband, unmarried, etc)
 - `race`: categorical race of the individual
 - `gender`: categorical gender of the individual
 - `capital-gain`: continuous variable measuring profit
 - `capital-loss`: continuous variable measuring loss
 - `hours-per-week`: the number of hours you work in a week
 - `native-country`: country of origin
 - `income`: whether your income is above or below/including $50,000