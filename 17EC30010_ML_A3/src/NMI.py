#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np


# In[18]:


from collections import Counter


# In[23]:


def NMI(List):
    """This Function finds NMI score by firstly calculating H(Y) then calling functions to find H(c) and H(Y|c) 
    and finding MI,I(Y;C)=H(Y)-H(Y|c) after calculating this it returns the NMI """
    data=pd.read_csv('data.csv',index_col=0)
    n_rows=data.shape[0]
    h_y=0
    for key,val in Counter(data.index).items():
        h_y-=(val/n_rows)*np.log2(val/n_rows)
    h_c=EntropyOfCluster(List,n_rows)
    MI=h_y-conditionalEntropy(List,n_rows,data)
    return (2*MI)/(h_y+h_c)


# In[33]:


def EntropyOfCluster(List,n_rows):
    """This Function finds the entropy H(c) """
    h_c=0
    for smallList in List:
        val=len(smallList)
        #print(val)
        if(val!=0):
            h_c-=(val/n_rows)*np.log2(val/n_rows)
    return h_c


# In[31]:


def conditionalEntropy(List,n_rows,data):
    """ This Function finds H(Y|c)"""
    CondEntpy=0
    for smallList in List:
        entpy=0
        Counts=Counter(data.index[smallList])
        length=len(smallList)
        for books in data.index.unique():
            count=Counts[books]
            if(count!=0):
                entpy-=(count/n_rows)*np.log2(count/length)
        CondEntpy+=entpy
    return CondEntpy


# In[ ]:




