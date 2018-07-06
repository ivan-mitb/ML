
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



# In[9]:


x_train_pca=x_train.iloc[:,4: ]
x_train_pca.head()


# In[10]:


mm_scale_train = preprocessing.MinMaxScaler().fit(x_train_pca)
x_train_pca_mm = mm_scale_train.transform(x_train_pca)


# In[11]:


pca=decomposition.PCA()
pca_result=pca.fit_transform(x_train_pca_mm)
pca.explained_variance_ratio_


# In[15]:


x_train_pca['attack_type'] = y_train.attack_type
x_train_pca['attack_type'] = x_train_pca['attack_type'].apply(lambda i: str(i))


# In[16]:


x_train_pca['pca-one'] = pca_result[:,0]
x_train_pca['pca-two'] = pca_result[:,1] 
x_train_pca['pca-three'] = pca_result[:,2]
x_train_pca['pca-four'] = pca_result[:,3]
x_train_pca['pca-five'] = pca_result[:,4]


# In[19]:


x_train_pca.shape


# In[17]:


from ggplot import *

chart = ggplot( x_train_pca, aes(x='pca-one', y='pca-two', color='attack_type') )         + geom_point(size=75,alpha=0.8)         + ggtitle("First and Second Principal Components colored by attack type")
chart


# In[18]:


pca.components_


