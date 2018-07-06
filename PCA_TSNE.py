
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
from sklearn import metrics,naive_bayes
from sklearn import preprocessing
from sklearn import tree
from sklearn import decomposition
import matplotlib.pyplot as plt
from ggplot import *
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D



# In[12]:


#from sklearn.decomposition import  KernelPCA


# In[26]:


#kpca = KernelPCA(kernel="rbf", gamma=15)
#Kpca_result= kpca.fit_transform(df_pca_mm)


# In[6]:


# the processed dataset is now in DataFrame 'df'.
# we first split it into train/test, before doing any analysis
#init_dataset(filename='C:/Users/Admin/Downloads/kdd99-data-lite/kddcup.data.txt')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df.iloc[:, -2:], test_size=0.1, random_state=4129)

# the dataset is now ready, in DataFrames x_train and x_test

save_object([x_train, y_train], 'train.dat')
save_object([x_test, y_test], 'test.dat')


# In[7]:

#creating arandom sample of 100K rows to fit the TSNE
np.random.seed(2018)
sample = np.random.choice([True, False], 100000, replace=True, p=[0.5,0.5])
x_sample = x_train.iloc[sample,22:-2]
y_sample = y_train.iloc[sample,-2:]


# In[8]:


mm_scale_train = preprocessing.MinMaxScaler().fit(x_sample)
x_sample_mm = mm_scale_train.transform(x_sample)


# In[60]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(x_sample_mm)


# In[61]:


x_sample['attack_type'] = y_sample.attack_type
x_sample['attack_type'] = x_sample['attack_type'].apply(lambda i: str(i))


# In[62]:


x_sample['x-tsne'] = tsne_results[:,0]
x_sample['y-tsne'] = tsne_results[:,1]

chart = ggplot( x_sample, aes(x='x-tsne', y='y-tsne', color='attack_type') )         + geom_point(size=500,alpha=0.1)         + ggtitle("tSNE dimensions colored by attack type")
chart


# In[20]:


x_train_pca=x_train.iloc[:,22:-2]
x_train_pca.head()


# In[21]:


mm_scale_train = preprocessing.MinMaxScaler().fit(x_train_pca)
x_train_pca_mm = mm_scale_train.transform(x_train_pca)


# In[22]:


pca=decomposition.PCA()
pca_result=pca.fit_transform(x_train_pca_mm)
pca.explained_variance_ratio_


# In[23]:


x_train_pca['attack_type'] = y_train.attack_type
x_train_pca['attack_type'] = x_train_pca['attack_type'].apply(lambda i: str(i))


# In[24]:


x_train_pca['pca-one'] = pca_result[:,0]
x_train_pca['pca-two'] = pca_result[:,1] 
x_train_pca['pca-three'] = pca_result[:,2]
x_train_pca['pca-four'] = pca_result[:,3]


# In[25]:


from ggplot import *

chart = ggplot( x_train_pca, aes(x='pca-one', y='pca-two', color='attack_type') )         + geom_point(size=75,alpha=0.8)         + ggtitle("First and Second Principal Components colored by attack type")
chart



