# Machine Learning Algorithms Implementation
This repository contains implemenstion of Machine Learning Algorithms like Linear Regression, Lasso Regression, Logistic Regression, ID3 Decision Tree algorithm, Agglomerative Clustering Algorithm and K-means Clustering Algorithm
### Assignment 1:
Contains implementation for Linear Regression, Lasso Regression and Rigde Regression. Linear Regression and Ridge Regression are implemented using Gradient descent whereas Lasso Regression is implemented using Coordinate descent algorithm. 
### Assignment 2:
Contains implementation for Logistic Regression and ID3 decision tree algorithm for multi-class classification. After implementing both the algorithms 3-fold-cross validation is performed and mean accuracy, precision and recall (mean-macro accuracy, mean-macro precision and mean-macro recall in case of decision tree multi class classifier) are calculated and compared with the implementations of scikit-learn package.
### Assignment 3:
1. Converted the Document Term Matrix (DTM) to TF-IDF matrix. Normalized the TF-IDF matrix by using L2 norm.
2. Implemented Agglomerative Clustering algorithm for obtaining 8 clusters of documents from Religious text dataset. Used single linkage strategy to join clusters.
3. Implemented K-means clustering algorithm to obtin k=8 clusters.(Used cosine similarity in both K-means and Agglomerative Clustering Algorithm). 
4. Reduced the counts of attribute to 100 using Principal Component Analysis (for this part used implementation of PCA from scikit-learn library.
5. Again used Agglomerative Clustering algorithm and K-means algorithm to find the clusters using reduced set of features.
