#%%Importing Libraries
import pandas as pd
import numpy as np
from collections import Counter
from pickle import copy 
#%% Loading the Relevant data
winedata=pd.read_csv('./dataB.csv')
data=winedata.to_numpy()
x_train=winedata.iloc[:,:-1].to_numpy()
y_train=winedata.iloc[:,-1].to_numpy()
#%% Node Class for Building Decision Tree
class Node:
    def __init__(self,attr=None,ch=None,y=None):
        self.attr=attr
        #attr is the attribute upon which the split is performed
        #if attr is none, it means that curr Node is not being selected and it is a leaf node
        self.y=y;
        #y is majority class of the node or prediction if given node is leaf node
        if ch is not None:
            self.ch=copy.deepcopy(ch)
        else:
            self.ch=[None,None,None,None]
        #ch are children nodes of which we get after splitting the data according to current attribute

#%% Information gain Calculation
def info_gain(data,split_attr):
    split_entpy=0
    N=data.shape[0];
    counts=Counter(data[:,split_attr])
    if len(counts)==1: return -1
    for attr_val,Ni in counts.items():
        class_counts=Counter(data[data[:,split_attr]==attr_val][:,-1])
        #print(class_counts)
        for class_values,nj in class_counts.items():
            split_entpy += nj*np.log2(Ni/nj)
    par_entpy=0
    par_counts=Counter(data[:,-1])
    for j,nj in par_counts.items():
        par_entpy+=nj*np.log2(N/nj)
    gain=(par_entpy-split_entpy)/N
    return gain    
#%%Decision Tree Building function
attrs_to_check=set(range(data.shape[1]-1))
def build_dt(data,parent_majority_class):
    if data.shape[0]==0:
        leaf=Node(None,None,parent_majority_class)
        return leaf
    elif data.shape[0]<10 or len(attrs_to_check)==0:
        max_cnt_class=next(iter(Counter(data[:,-1]).most_common(1)))[0]
        leaf=Node(None,None,max_cnt_class)
        return leaf
    else:
        classes=Counter(data[:,-1])
        if(len(classes)==1):
            class_val=data[0,-1]
            leaf=Node(None,None,class_val)
            return leaf
        max_gain=0
        attr_pick=None
        for attr in attrs_to_check:
            curr_gain=info_gain(data,attr)
            if curr_gain<0:
                continue
            if(curr_gain>max_gain):
                max_gain=curr_gain
                attr_pick=attr
        max_cnt_class=next(iter(classes.most_common(1)))[0]
        if attr_pick is None:
            leaf=Node(None,None,max_cnt_class)
            return leaf
        else:
            curr_node=Node(attr_pick,None,max_cnt_class)
            attrs_to_check.remove(attr_pick)
            for i in range(4):
                curr_node.ch[i]=build_dt(data[data[:,attr_pick]==i],max_cnt_class)
            attrs_to_check.add(attr_pick)
            return curr_node
#%% Prediction function for decision Tree
def predict_dt(node,x):
    if node.attr is None:
        return node.y
    attr_val=x[node.attr]
    return predict_dt(node.ch[attr_val],x)
#%% Building DT
root=build_dt(data,0)
#%% Calculating DT predictions
from sklearn.metrics import classification_report
correct_predictions=0
predictions=np.zeros(data.shape[0])
for i in range(data.shape[0]):
    y_predict=predict_dt(root,data[i,:-1])
    if y_predict==data[i,-1]:
        correct_predictions+=1
    predictions[i]=y_predict
print("train-metrics for our DT implementation")
print(classification_report(data[:,-1],predictions))
#%% K-Fold Cross Validation

from sklearn.model_selection import KFold
kfold=KFold(3,True,1)
conf_mat=4*np.ones((3,3))
predictionsKfold=np.zeros(data.shape[0])
#calculating confusion matrix
for train,test in kfold.split(data):
    train_data,test_data=data[train],data[test]
    root=build_dt(train_data,0)
    for i in range(test_data.shape[0]):
        y_predict=predict_dt(root,test_data[i,:-1])
        predictionsKfold[test[i]]=y_predict
        conf_mat[data[test[i],-1],y_predict]+=1
#%% Calculating Macro Precision, Recall and accuracy
pos_pred_cnt=np.sum(conf_mat,0)
pos_actual_cnt=np.sum(conf_mat,1)
tot_sum=np.sum(conf_mat)
precision=0
recall=0
accuracy=0
for i in range(3):
    precision+=conf_mat[i,i]/pos_pred_cnt[i]
    recall+=conf_mat[i,i]/pos_actual_cnt[i]
    accuracy+=conf_mat[i,i]
accuracy/=tot_sum
macro_precision=precision/3
macro_accuracy=accuracy
macro_recall=recall/3
print(f"macro accuracy = {macro_accuracy}\nmacro precision = {macro_precision}\nmacro recall= {macro_recall}")
print("for our decision tree implementation using 3-fold cross validation\n")
print(classification_report(data[:,-1],predictionsKfold))
#%%sk learn implementation of decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy",min_samples_split=10)
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_train)
from sklearn.metrics import  classification_report
print("train metrics for sk-learn decision tree\n")
print(classification_report(y_train,dt_predict))
#%%k-fold cross validation of sk-learn decision tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
dt1=DecisionTreeClassifier(criterion="entropy",min_samples_split=10)
dt_predict_cv=cross_val_predict(dt1,x_train,y_train,cv=3)
print("3-fold cross-validation metrics of sk-learn DT\n")
print(classification_report(y_train,dt_predict_cv))
