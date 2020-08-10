#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sklearn
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[29]:


#preprocessing winedata for using logistic regression
#winedata_up is just the raw data with only min-max scaling
winedata=pd.read_csv("dataA.csv")


# In[30]:


op=winedata['quality']
op_new=op.to_numpy()


# In[31]:


ip_new=winedata.drop('quality',axis=1)
ip_new=ip_new.to_numpy()


# In[32]:


#sigmoid function for logistic regression
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
sigmoid_vec = np.vectorize(sigmoid)


# In[35]:


#Our Logistic Regression Implementation
def logistic_regression(ip_new,op_new):
    ip=np.zeros((ip_new.shape[0],ip_new.shape[1]+1))
    ip[:,0]=1
    ip[:,1:]=ip_new
    op=op_new.reshape((op_new.shape[0],1))
    thetas=np.zeros((ip.shape[1],1))
    h_theta=np.zeros((ip.shape[0],1))
    errors=np.zeros((ip.shape[0],1))
    alpha=10
    m=ip.shape[0]
    while(True):
        h_theta=sigmoid_vec(ip@thetas)
        errors=h_theta-op
        partial=(1/m)*np.matmul(ip.transpose(),errors)
        if(np.sum(partial**2)**.5<.00001):
            break
        thetas=thetas-alpha*partial
    return thetas


# In[36]:


#Function to predict the classes given the parameters of log_reg
def lr_predict(ip_new,thetas):
    ip=np.zeros((ip_new.shape[0],ip_new.shape[1]+1))
    ip[:,0]=1
    ip[:,1:]=ip_new
    p_y=sigmoid_vec(ip@thetas)
    p_y=(p_y>=.5)*1
    return p_y


# In[37]:


thetas=logistic_regression(ip_new,op_new)
p_y=lr_predict(ip_new,thetas)


# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[40]:


cf1=confusion_matrix(op_new,p_y,labels=[1,0])
acs=accuracy_score(op_new,p_y)
precision=cf1[0,0]/(np.sum(cf1[:,0]))
recall=cf1[0,0]/(np.sum(cf1[0,:]))
print('metrics for test set\n')
print(f'for our implementation for training set precision is {precision} recall is {recall} accuracy is {acs}\n')


# In[42]:


#saga solver for sklearn
lr = LogisticRegression(solver='saga',max_iter=10000,penalty='none',tol=.000001)
lr.fit(ip_new,op_new)


# In[43]:


p_y_skl = lr.predict(ip_new)


# In[44]:


cf_1=confusion_matrix(op_new,p_y_skl,labels=[1,0])
acs=accuracy_score(op_new,p_y_skl)
precision=cf_1[0,0]/(np.sum(cf_1[:,0]))
recall=cf_1[0,0]/(np.sum(cf_1[0,:]))
print(f'for skl implementation for training set precision is {precision} recall is {recall} accuracy is {acs}\n')


# In[45]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
scores=np.zeros((3,3))
i=0
for train_index, test_index in kf.split(ip_new):
    #print("TRAIN:", train_index, "TEST:", test_index)
    ip_train, ip_test = ip_new[train_index], ip_new[test_index]
    op_train, op_test = op_new[train_index], op_new[test_index]
    thetas=logistic_regression(ip_train,op_train)
    op_predict=lr_predict(ip_test,thetas)
    cf2=confusion_matrix(op_test,op_predict,labels=[1,0])
    acs=accuracy_score(op_test,op_predict)
    precision=cf2[0,0]/(np.sum(cf2[:,0]))
    recall=cf2[0,0]/(np.sum(cf2[0,:]))
    scores[i,0]=precision
    scores[i,1]=recall
    scores[i,2]=acs
    i+=1


# In[46]:

print('metrics for 3-fold cross validation\n')
mean_scores1=np.mean(scores,axis=0)
print(f'for our implementation precision is {mean_scores1[0]} recall is {mean_scores1[1]} accuracy is {mean_scores1[2]}\n')


# In[47]:


from sklearn.model_selection import cross_val_predict
logreg=LogisticRegression(solver='saga',max_iter=10000,penalty='none',tol=.000001)
op_predict=cross_val_predict(logreg,ip_new, op_new, cv=3)
cf_2=confusion_matrix(op_new,op_predict,labels=[1,0])
acs2=accuracy_score(op_new,op_predict)
precision2=cf_2[0,0]/(np.sum(cf_2[:,0]))
recall2=cf_2[0,0]/(np.sum(cf_2[0,:]))
print(f'for skl implementation precision is {precision2} recall is {recall2} accuracy is {acs2}')


# In[ ]:




