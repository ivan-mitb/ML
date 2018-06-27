# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 16:12:45 2018

@author: Nic
"""
from sklearn import preprocessing
from dataload import load_object, save_object
import pandas as pd
import numpy as np

x_train, y_train = load_object('train.dat')

### convert categorical to dummy binaries
x_train_dum = x_train.copy()
to_convert_dum = ['protocol_type', 'service', 'flag']
tmp_df = []
for i in range(len(to_convert_dum)):
    tmp_df.append(pd.get_dummies(x_train.iloc[:, x_train.columns.get_loc(to_convert_dum[i])]))
    x_train_dum = x_train_dum.drop([to_convert_dum[i]], axis = 1)
    x_train_dum = pd.concat([x_train_dum, tmp_df[i]], axis = 1)

### get list to be scaled
scale_list = []
for i in range(x_train_dum.shape[1]):
    if (max(x_train_dum.iloc[:, i]) > 9):
        print(i, ". ", x_train_dum.columns.values[i], ' - Max : ', max(x_train_dum.iloc[:, i]))
        scale_list.append(i)
### remove discrete columns max value are natually below 9 so will not be included
        
### Min-Max Scalar
mms = preprocessing.MinMaxScaler()
x_train_mms = x_train_dum.copy()
for i in range(len(scale_list)):
    print(i)
    x_train_mms.iloc[:, [scale_list[i]]] = mms.fit_transform(x_train_dum.iloc[:, [scale_list[i]]])
    
##print to compare min-max before and after scaling
for i in range(x_train_dum.shape[1]):
    print(i, " : Scaled>> ", min(x_train_mms.iloc[:, i]), '-', max(x_train_mms.iloc[:, i]), 
          ', UnScaled>>', min(x_train_dum.iloc[:, i]), '-', max(x_train_dum.iloc[:, i]))

### Binning
x_train_bin = x_train_mms.copy()
for i in range(len(scale_list)):
    print(i)
    bins = np.array([0.0, 
                     np.percentile(x_train_mms.iloc[:, [scale_list[i]]], 25),
                     np.percentile(x_train_mms.iloc[:, [scale_list[i]]], 50),
                     np.percentile(x_train_mms.iloc[:, [scale_list[i]]], 75)
                     ])
    x_train_bin.iloc[:, [scale_list[i]]] = np.digitize(x_train_mms.iloc[:, [scale_list[i]]], bins)

### Save to object
save_object([x_train_bin, y_train], 'train_bin.dat')
### Save to view
np.savetxt("x_train_bin.csv", x_train_bin, delimiter=",")

'''
### Robust Scalar
rs = preprocessing.RobustScaler()
x_train_rs = x_train_dum.copy()
for i in range(len(scale_list)):
    x_train_qt.iloc[:, [scale_list[i]]] = rs.fit_transform(x_train_dum.iloc[:, [scale_list[i]]])
np.savetxt("x_train_rs.csv", x_train_rs, delimiter=",")

### Quantile Transformer
qt = preprocessing.QuantileTransformer(output_distribution='normal')
x_train_qt = x_train_dum.copy()
for i in range(len(scale_list)):
    x_train_qt.iloc[:, [scale_list[i]]] = qt.fit_transform(x_train_dum.iloc[:, [scale_list[i]]])
np.savetxt("x_train_qt.csv", x_train_qt, delimiter=",")

### Log Transformer
logtf = preprocessing.FunctionTransformer(np.log1p)
x_train_logtf = x_train_dum.copy()
for i in range(len(scale_list)):
    x_train_qt.iloc[:, [scale_list[i]]] = logtf.transform(x_train_dum.iloc[:, [scale_list[i]]])
np.savetxt("x_train_logtf.csv", x_train_logtf, delimiter=",")
'''

