#!/usr/bin/env python
# coding: utf-8

# In[278]:


import numpy as np
import pandas as pd


# In[279]:


df=pd.read_csv('labelled.csv')


# In[280]:


df.drop(index=13,inplace=True)


# In[281]:


df.iloc[:,0]=df.iloc[:,0].apply(lambda x:x.split("_")[0])


# In[282]:


df=df.rename(index=df.iloc[:,0])


# In[283]:


df=df.drop(columns='Unnamed: 0')


# In[284]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[285]:


df_copy=df.copy()


# In[286]:


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)


# In[287]:


tfidf_transformer.fit(df_copy.to_numpy())


# In[288]:


idf=tfidf_transformer.idf_-1


# In[289]:


idf


# In[290]:


df_copy=df*idf


# In[291]:


df


# In[292]:


df_copy


# In[293]:


myfunc=lambda row:row.apply(lambda x:x**2).sum()**.5


# In[294]:


df_norm=df_copy.apply(myfunc,axis=1).to_numpy()


# In[296]:


df_normalized=(df_copy.T/df_norm).T


# In[297]:


df_normalized.to_csv('data.csv')

