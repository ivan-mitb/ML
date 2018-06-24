# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 16:12:45 2018

@author: Nic
"""
from sklearn import preprocessing
from dataload import load_object, save_object
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

x_train_r, y_train_r = load_object('train_r.dat')
scale_list = []
for i in range(x_train_r.shape[1]):
    if (max(x_train_r[:, i]) > 9):
#        print(i, ' - Max : ', max(x_train_r[:, i]))
        scale_list.append(i)
### remove discrete columns
#scale_list.remove(2)
#scale_list.remove(7)
#scale_list.remove(9)
#scale_list.remove(10)
#scale_list.remove(16)
#scale_list.remove(17)
        
### Min-Max Scalar
mms = preprocessing.MinMaxScaler()
x_train_r_mms = x_train_r.copy()
for i in range(len(scale_list)):
    x_train_r_mms[:, [scale_list[i]]] = mms.fit_transform(x_train_r[:, [scale_list[i]]])
##print to compare min-max before and after scaling
#for i in range(x_train_r.shape[1]):
#    print(i, " : Scaled>> ", min(x_train_r_scaled[:, i]), '-', max(x_train_r_scaled[:, i]), 
#          ', UnScaled>>', min(x_train_r[:, i]), '-', max(x_train_r[:, i]))
np.savetxt("x_train_r_mms.csv", x_train_r_mms, delimiter=",")
### Binning
x_train_r_mms_bin = x_train_r_mms.copy()
for i in range(len(scale_list)):
    bins = np.array([0.0, 
                     np.percentile(x_train_r_mms[:, [scale_list[i]]], 25),
                     np.percentile(x_train_r_mms[:, [scale_list[i]]], 50),
                     np.percentile(x_train_r_mms[:, [scale_list[i]]], 75)
                     ])
    x_train_r_mms_bin[:, [scale_list[i]]] = np.digitize(x_train_r_mms[:, [scale_list[i]]], bins)
np.savetxt("x_train_r_mms_bin.csv", x_train_r_mms_bin, delimiter=",")

'''
x_train_r_mms_df = pd.DataFrame(x_train_r_mms)
plt.hist(x_train_r_mms_df.iloc[:,0], bins = x_train_r_mms_df.iloc[:,0].value_counts().shape[0])
### Robust Scalar
rs = preprocessing.RobustScaler()
x_train_r_rs = x_train_r.copy()
for i in range(len(scale_list)):
    x_train_r_rs[:, [scale_list[i]]] = rs.fit_transform(x_train_r[:, [scale_list[i]]])
np.savetxt("x_train_r_rs.csv", x_train_r_rs, delimiter=",")

### Quantile Transformer
qt = preprocessing.QuantileTransformer(output_distribution='normal')
x_train_r_qt = x_train_r.copy()
for i in range(len(scale_list)):
    x_train_r_qt[:, [scale_list[i]]] = qt.fit_transform(x_train_r[:, [scale_list[i]]])
np.savetxt("x_train_r_qt.csv", x_train_r_qt, delimiter=",")

### Log Transformer
logtf = preprocessing.FunctionTransformer(np.log1p)
x_train_r_logtf = x_train_r.copy()
for i in range(len(scale_list)):
    x_train_r_qt[:, [scale_list[i]]] = logtf.transform(x_train_r[:, [scale_list[i]]])
np.savetxt("x_train_r_logtf.csv", x_train_r_logtf, delimiter=",")
'''

