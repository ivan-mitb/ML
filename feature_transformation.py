# -*- coding: utf-6 -*-
"""
Created on Sun Jun 24 16:12:45 2018

@author: Nic
"""
from sklearn import preprocessing
from dataload import load_object, save_object
import pandas as pd
import numpy as np

def cat2dummy(x_train, cols=['protocol_type', 'service', 'flag']):
    ''' convert categorical to dummy binaries '''
    x_train_dum = x_train.copy()
    to_convert_dum = cols
    tmp_df = []
    for i in range(len(to_convert_dum)):
        tmp_df.append(pd.get_dummies(x_train.iloc[:, x_train.columns.get_loc(to_convert_dum[i])]))
        x_train_dum = x_train_dum.drop([to_convert_dum[i]], axis = 1)
        x_train_dum = pd.concat([x_train_dum, tmp_df[i]], axis = 1)
    return x_train_dum

def scale_list(x_train_dum):
    ''' returns a list of col indices to be scaled '''
    scale_list = []
    for i in range(x_train_dum.shape[1]):
        if (max(x_train_dum.iloc[:, i]) > 9):
            print(i, ". ", x_train_dum.columns.values[i], ' - Max : ', max(x_train_dum.iloc[:, i]))
            scale_list.append(i)
    ### remove discrete columns max value are natually below 9 so will not be included
    return scale_list

def minmax(x_train_dum, scale_list):
    ''' Min-Max Scaler '''
    mms = preprocessing.MinMaxScaler()
    x_train_mms = x_train_dum.copy()
    for i in scale_list:
        print(i)
        x_train_mms.iloc[:, i] = mms.fit_transform(x_train_dum.iloc[:, i])
    return x_train_mms

def binning(x_train_mms, scale_list):
    ''' Binning by quartiles '''
    x_train_bin = x_train_mms.copy()
    for i in scale_list:
        print(i)
        bins = np.array([0.0,
                         np.percentile(x_train_mms.iloc[:, i], 25),
                         np.percentile(x_train_mms.iloc[:, i], 50),
                         np.percentile(x_train_mms.iloc[:, i], 75)
                         ])
        x_train_bin.iloc[:, i] = np.digitize(x_train_mms.iloc[:, i], bins)
    return x_train_bin

def robust(x_train_dum, scale_list):
    ''' Robust Scaler '''
    rs = preprocessing.RobustScaler()
    x_train_rs = x_train_dum.copy()
    for i in scale_list:
        x_train_rs.iloc[:, i] = rs.fit_transform(x_train_dum.iloc[:, i])
    return x_train_rs

def quantile(x_train_dum, scale_list):
    ''' Quantile Transformer '''
    qt = preprocessing.QuantileTransformer(output_distribution='normal')
    x_train_qt = x_train_dum.copy()
    for i in scale_list:
        x_train_qt.iloc[:, i] = qt.fit_transform(x_train_dum.iloc[:, i])
    return x_train_qt

def log_transform(x_train_dum, scale_list):
    ''' Log Transformer '''
    logtf = preprocessing.FunctionTransformer(np.log1p)
    x_train_logtf = x_train_dum.copy()
    for i in scale_list:
        x_train_logtf.iloc[:, i] = logtf.transform(x_train_dum.iloc[:, i])
    return x_train_logtf
