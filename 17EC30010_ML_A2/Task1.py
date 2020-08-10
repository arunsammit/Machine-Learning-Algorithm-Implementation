#!/usr/bin/env python
# coding: utf-8

# In[166]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[167]:


winedata=pd.read_csv("winequality-red.csv",sep=';')
winedata['quality']=(winedata['quality']>6)*1
winedata.iloc[:,:-1] = scaler.fit_transform(winedata.iloc[:,:-1])
winedata.to_csv('dataA.csv',index=False)


# In[168]:


def quality(q):
    
    if q<5 :
        return 0
    elif q>6 :
        return 2
    else:
        return 1


# In[169]:


winedata=pd.read_csv("winequality-red.csv",sep=';')
winedata['quality']=winedata['quality'].apply(quality)


# In[170]:


winedata.iloc[:,:-1]-=winedata.iloc[:,:-1].mean(axis=0)
winedata.iloc[:,:-1]/=winedata.iloc[:,:-1].std(axis=0,ddof=0)


# In[172]:


small=winedata.iloc[:,:-1].min()
big=winedata.iloc[:,:-1].max()
bins=[np.linspace(s,b,5) for s,b in zip(small,big)]


# In[173]:


def binning(elem,arr):
    for i in range(4):
        if((elem<=arr[i+1]) and (elem>=arr[i])):
            return i


# In[174]:


for i in range(winedata.shape[1]-1):
    winedata.iloc[:,i]=winedata.iloc[:,i].apply(binning,args=(bins[i],))


# In[175]:


winedata.to_csv('dataB.csv',index=False)

