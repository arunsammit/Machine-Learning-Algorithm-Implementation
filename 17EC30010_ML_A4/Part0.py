import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import zscore

data = np.loadtxt("dataset.txt") 

# Spliting of Data into Train & Test
train, test = train_test_split(data, test_size = 0.2)
print(test)
# Z score normalization of features
train[:,:-1] = zscore(train[:,:-1], axis=1)
test[:,:-1] = zscore(test[:,:-1], axis=1)
print(test)
#%%
# One hot representation of labels
#onehotencoder = OneHotEncoder(categorical_features = [0])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [7])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
train1=np.array(onehotencoder.fit_transform(train),dtype=np.float)
test1=np.array(onehotencoder.fit_transform(test),dtype=np.float)
train2=np.concatenate((train1[:,3:],train1[:,:3]),axis=1)
test2=np.concatenate((test1[:,3:],test1[:,:3]),axis=1)
#%%
# Saving Data Files
np.savetxt("train.txt", train2)
np.savetxt("test.txt", test2)