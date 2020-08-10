#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np 
import pandas as pd
import numpy.linalg as la


# In[51]:


data1=pd.read_csv('data.csv',index_col=0)
data=data1.to_numpy()
data2=pd.read_csv('data_reduced.csv',index_col=0)
data_reduced=data2.to_numpy()


# In[54]:


def cosine_sim(vect1,vect2):
    return np.dot(vect1,vect2)/(la.norm(vect1)*la.norm(vect2))


# In[55]:


def kmeans(data,k):
    n = data.shape[0]
    c = data.shape[1]
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    centers = np.random.randn(k,c)*std + mean

    centers_old = np.zeros(centers.shape)
    centers_new = np.copy(centers)
    clusters = np.zeros(n)
    similarities = np.zeros((n,k))
    error = la.norm(centers_new - centers_old)

    while error != 0:
        for i in range(k):
            similarities[:,i]=np.apply_along_axis(lambda row:cosine_sim(row,centers[i]),1,data)
        clusters = np.argmax(similarities, axis = 1)
        #print(clusters)
        centers_old = np.copy(centers_new)
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = la.norm(centers_new - centers_old)
    cluster_list=[[i for i,val in enumerate(clusters) if val==j] for j in range(0,8)]
    cluster_list=list(map(lambda x:sorted(x),cluster_list))
    cluster_list.sort(key=lambda x:x[0])
    return cluster_list


# In[56]:


def savefile(cluster_list,fname): 
    f=open(fname,'w')
    for i in range(0,8):
        cluster=cluster_list[i]
        l=len(cluster)
        for j in range(0,l):
            f.write(str(cluster[j]))
            if(j<l-1):
                f.write(',')
        f.write('\n')
    f.close()


# In[57]:


cluster_list=kmeans(data,8)
cluster_list_from_reduced=kmeans(data_reduced,8)


# In[58]:


savefile(cluster_list,'kmeans.txt')
savefile(cluster_list_from_reduced,'kmeans_reduced.txt')


# In[61]:


from NMI import NMI


# In[62]:


NMIkmeans=NMI(cluster_list)
print(f"NMI score for kmeans clustering is {NMIkmeans}")


# In[63]:


NMIkmeans_reduced=NMI(cluster_list_from_reduced)
print(f"NMI score for kmeans clustering using reduced dataset is {NMIkmeans_reduced}")


# In[ ]:




