INSTRUCTIONS TO EXECUTE/RUN THE CODES
1. Copy labelled.csv from data folder to src folder
2. Run PartA.py to ger data.csv which is the dataset prepared according to Question A
3. Run PCA.py to get data_reduced.csv which is the reduced dataset with 100 features/attributes as asked in Question D
4. Run PartB.py which is the solution to Agglomerative Clustering problem aksed in Question B. It will do Agglomerative clustering of dataset prepared 
in part A and of data set with reduced features.It will save the obtained clusters in 'agglomerative.txt' and 'agglomerative_reduced.txt'.
It will also print the NMI scores of the two clusters obtained by importing NMI function from file 'NMI.py' in which I have written the required functions.
5. Simililarly, run PartC.py to get all the results for Kmeans clustering.
6. Solution for Question E is contained in file 'NMI.py'
NMI score
Agglomerative Clustering:.0245
Agglomerative Clustering with reduced features: .0245

Kmeans: .0878**
Kmeans with reduced feature: .2279**

** Note that these scores may vary when you run 'PartC.py' file because centers for clusters are initialised randomly and the objective function of K means is not convex