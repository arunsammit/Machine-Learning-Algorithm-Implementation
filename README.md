# Machine Learning Algorithms Implementation
## TLDR
1. Built an n-Layer MLP classifier to classify seeds into 3 classes from scratch using Mini Batch Stochastic Gradient Descent. Got test accuracy of 90% with our implementation and 95% with MLP Classifier of sk-learn library.
2. Implemented Agglomerative and K-means Clustering Algorithms for clustering text documents using tf-idf to get
NMI scores of .0245 and .0878 respectively. Used PCA to reduce the features to 100, which improved the NMI score
of K-means to .2279.
3. Implemented the ID3 Decision tree algorithm to classify wine-quality into three classes. Performed 3-fold Cross-
Validation to get Macro-F1 score of .5 for our implementation and .45 for sklearn implementation.
4. Implemented Lasso Regression using Coordinate descent and Ridge Regression using Gradient descent for fitting a
polynomial curve of any degree to a given training data. Got .03 and .01 MSE on test data for them respectively.


This repository contains implemenstion of Machine Learning Algorithms like Linear Regression, Lasso Regression, Logistic Regression, ID3 Decision Tree algorithm, Agglomerative Clustering Algorithm, K-means Clustering Algorithm and L-Layer deep neural network for multi-class classification which are taught in Under-Graduate Machine Learning course at IIT Kharagpur
### Assignment 1:
Contains implementation for Linear Regression, Lasso Regression and Rigde Regression. Linear Regression and Ridge Regression are implemented using Gradient descent whereas Lasso Regression is implemented using Coordinate descent algorithm. 
### Assignment 2:
Contains implementation for Logistic Regression and ID3 decision tree algorithm for multi-class classification. After implementing both the algorithms 3-fold-cross validation is performed and mean accuracy, precision and recall (mean-macro accuracy, mean-macro precision and mean-macro recall in case of decision tree multi class classifier) are calculated and compared with the implementations of scikit-learn package.
### Assignment 3:
1. Converted the Document Term Matrix (DTM) to TF-IDF matrix. Normalized the TF-IDF matrix by using L2 norm.
2. Implemented Agglomerative Clustering algorithm for obtaining 8 clusters of documents from Religious text dataset. Used single linkage strategy to join clusters.
3. Implemented K-means clustering algorithm to obtin k=8 clusters.(Used cosine similarity in both K-means and Agglomerative Clustering Algorithm). 
4. Reduced the counts of attribute to 100 using Principal Component Analysis (for this part used implementation of PCA from scikit-learn library).
5. Again used Agglomerative Clustering algorithm and K-means algorithm to find the clusters using reduced set of features.
### Assignment 4:
Contains the implementation for L-layer deep Neural Network for multi class classification. Optimisation algorithm used: Mini Batch Gradient Descent (SGD), Loss function used: categorical entropy loss.
