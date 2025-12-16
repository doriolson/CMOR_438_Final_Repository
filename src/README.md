## The Source Code for Rice ML

This folder contains the source code for all of our algorithms within the repository. We use the src primarily as a location to store the code for all of the functions which we will call in our example folders. Doing this ensures that the example code is as clean and efficient as possible. 

The src folder also contains our processing functions separate from the algorithms themselves, in the processing folder. Thus, the src is divided into processing, supervised learning, and unsupervised learning. The files within each are in essence a directory of the functions needed to perform each machine learning algorithm. The __init__.py files found throughout this folder are there so that the functions created in this src folder can be called from other files and folders within the repository. This is the basis of how our example code will be formatted, the functions all exist in this src file and will be called wherever they are needed via init.

Additionally, the src folder serves as a place for us to work through the code and draft the code for various other folders.

## Files (and functions) contained in this folder

All of the files are contained within a machine learning folder, which is within the general source code folder.
 
Processing folder:
 - init
 - pre processing
 - post processing

Supervised Learning Folder
 - init
 - decision trees
 - distance metrics
 - ensemble methods
 - k nearest neighbors
 - linear regression
 - logistic regression
 - multilayer perceptron
 - perceptron
 - regression trees

Unsupervised Learning Folder
- init
- community detection
- dbscan
- k means clustering
- principal component analysis