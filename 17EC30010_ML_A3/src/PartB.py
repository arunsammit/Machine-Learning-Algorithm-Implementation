#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def distance(d1,d2):
    return np.exp(-1*np.dot(d1,d2))


# In[3]:


def cosine_sim(data1,data2):
    return np.dot(data1,data2)


# In[4]:


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


# In[5]:


def s_m(data):
    """This function returns the similarity matrix. It computes it only for j>i"""
    n=data.shape[0]
    s=np.zeros((n,n))
    for i in range(0,n):
        for j in range(i+1,n):
            s[i,j]=cosine_sim(data[i,:],data[j,:])    
    return s


# In[6]:


def agglomerative(data,n_clusters):
    n_rows=data.shape[0]
    cluster_group=np.zeros((n_rows,n_rows+1),dtype=int)
    for i in range(0,n_rows):
        cluster_group[i,0]=i
        cluster_group[i,-1]=1
    merged_up_clusters=[]
    sim_mat=s_m(data)
    similarities_secondary_cluster=np.zeros(n_rows)
    for n_iter in range(n_clusters,n_rows):
        secondaryCluster,MainCluster = np.unravel_index(sim_mat.argmax(), sim_mat.shape)
        n_1=cluster_group[MainCluster,-1]
        n_2=cluster_group[secondaryCluster,-1]
        for i in range(0,n_2):
            cluster_group[MainCluster,i+n_1]=cluster_group[secondaryCluster,i]
        cluster_group[MainCluster,-1]=n_1+n_2
        for i in range(0,secondaryCluster):
            similarities_secondary_cluster[i]=sim_mat[i,secondaryCluster]
            sim_mat[i,secondaryCluster]=0
        for j in range(secondaryCluster,n_rows):
            similarities_secondary_cluster[j]=sim_mat[secondaryCluster,j]
            sim_mat[secondaryCluster,j]=0
        for i in range(0,MainCluster):
            sim_mat[i,MainCluster]=max(sim_mat[i,MainCluster],similarities_secondary_cluster[i])
        for j in range(MainCluster+1,n_rows):
            sim_mat[MainCluster,j]=max(sim_mat[MainCluster,j],similarities_secondary_cluster[j])
        sim_mat[MainCluster,secondaryCluster]=0
        sim_mat[secondaryCluster,MainCluster]=0
        merged_up_clusters.append(secondaryCluster)
    cluster_group=np.delete(cluster_group,merged_up_clusters,axis=0)
    cluster_list_unsorted=[[val for idx,val in enumerate(cluster_group[i]) if (idx<cluster_group[i,-1])] for i in range(0,n_clusters)]
    cluster_list=list(map(lambda x:sorted(x),cluster_list_unsorted))
    cluster_list.sort(key=lambda x:x[0])
    return cluster_list


# In[7]:


data=pd.read_csv('data.csv',index_col=0)
reduced_data=pd.read_csv('data_reduced.csv',index_col=0)
data=data.to_numpy()
reduced_data=reduced_data.to_numpy()


# In[8]:


cluster_list=agglomerative(data,8)
cluster_list_from_reduced=agglomerative(reduced_data,8)


# In[14]:


savefile(cluster_list,'agglomerative.txt')
savefile(cluster_list_from_reduced,'agglomerative_reduced.txt')


# In[15]:


from NMI import NMI


# In[23]:


NMIagglomerative=NMI(cluster_list)
print(f"NMI score for agglomerative clustering is {NMIagglomerative}")


# In[22]:


NMIagglomerative_reduced=NMI(cluster_list_from_reduced)
print(f"NMI score for agglomerative clustering using reduced features is {NMIagglomerative}")


# In[ ]:




