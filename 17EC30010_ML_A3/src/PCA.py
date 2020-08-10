#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[6]:


data=pd.read_csv('data.csv',index_col=0)


# In[2]:


from sklearn.decomposition import PCA


# In[3]:


pca=PCA(n_components=100)


# In[8]:


principalComponents=pca.fit_transform(data)


# In[9]:


principalComponents


# In[10]:


cols=[f'pc{x}' for x in range(0,100)]


# In[15]:


principalDf=pd.DataFrame(data=principalComponents,columns=cols,index=data.index)


# In[17]:


principalDf.to_csv('data_reduced.csv')


# In[ ]:




