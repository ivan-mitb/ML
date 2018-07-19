# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 21:31:06 2018

@author: Nic
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

colors = ['red', 'green', 'blue', 'yellow', 'purple', 'deeppink']
y_train_r_dum = pd.get_dummies(y_train_r)

def create_roc_dict(fpr, tpr, roc_auc):
    return {'fpr': fpr, 'tpr': tpr, 'area': roc_auc}

def get_fpr_tpr(clf, x_test, y_test):
#    y_pred = pd.get_dummies(clf.predict(x_test))
    if (isinstance(clf, OneClassSVM)) or (isinstance(clf, SVC)):
        y_pred = pd.DataFrame(clf.decision_function(x_test))
    else:        
        y_pred = pd.DataFrame(clf.predict_proba(x_test))
    y_test_dum = pd.get_dummies(y_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(y_pred.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_dum.iloc[:, i], y_pred.iloc[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
#    print('tpr: ', tpr[0].shape[0], ', fpr: ', fpr[0].shape[0])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_dum.values.ravel(), y_pred.values.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_pred.shape[1])]))
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(y_pred.shape[1]):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= y_pred.shape[1]
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return create_roc_dict(fpr, tpr, roc_auc)

def plot_ROC_curve_types(roc_dict, attack_types, name):
    plt.figure(name, figsize=(16, 8))
    plt.suptitle('ROC curve of {0}'.format(name), fontsize=16)
    plt.subplot(121)
    lw = 2
    for i in range(len(attack_types)):
        plt.plot(roc_dict['fpr'][i], roc_dict['tpr'][i], color=colors[i], lw=lw,
                 label='ROC curve of {0} (area = {1:0.4f})'
                 ''.format(attack_types[i], roc_dict['area'][i]))
        
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    plt.subplot(122)
    for i in range(len(attack_types)):
        plt.plot(roc_dict['fpr'][i], roc_dict['tpr'][i], color=colors[i], lw=lw,
                 label=attack_types[i])
        
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 0.2])
    plt.ylim([0.8, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic (zoomed)')
    plt.title('Zoomed top left')
    plt.legend(loc="lower right")
    plt.show()

def plot_ROC_curve_classes(clf_dict):
    lw = 3
    plt.figure(figsize=(16, 8))
    plt.suptitle('ROC across models', fontsize=16)
    
    plt.subplot(121)
    for key in clf_dict:
        plt.plot(clf_dict[key]['fpr']["macro"], clf_dict[key]['tpr']["macro"], label='Average ROC curve of {0} (area = {1:0.4f})'.format(key, clf_dict[key]['area']["macro"]),
         color=colors[list(clf_dict.keys()).index(key)], linewidth=lw)  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    
    plt.subplot(122)
    for key in clf_dict:
        plt.plot(clf_dict[key]['fpr']["macro"], clf_dict[key]['tpr']["macro"], label='{0}'.format(key),
         color=colors[list(clf_dict.keys()).index(key)], linewidth=lw)  
    plt.title('Zoomed top left')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 0.4])
    plt.ylim([0.8, 1.01])
    plt.legend(loc="lower right")
    plt.show()
    
### Compare the ROC across each type for each classifier
rf1 = RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=19, max_features=9,
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=9, min_weight_fraction_leaf=0.0,
            n_estimators=51, n_jobs=-1, oob_score=False, random_state=4129,
            verbose=0, warm_start=False)
rf1.fit(x_train_r, y_train_r)
rf1_roc_dict = get_fpr_tpr(rf1, x_corr, y_corr.iloc[:, 1])
plot_ROC_curve_types(rf1_roc_dict, y_train_r_dum.columns.values, 'Random Forest Classifier')

gb1 = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=20,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=50,
              presort='auto', random_state=4129, subsample=1.0, verbose=1,
              warm_start=False)
gb1.fit(x_train_r, y_train_r)
gb1_roc_dict = get_fpr_tpr(gb1, x_corr, y_corr.iloc[:, 1])
plot_ROC_curve_types(gb1_roc_dict, y_train_r_dum.columns.values, 'Gradient Boosting Classifier')

clf_dict = {}
clf_dict['rf1'] = rf1_roc_dict
clf_dict['gb1'] = gb1_roc_dict
plot_ROC_curve_classes(clf_dict)


### Compare across ROC curve Classifiers###################################
gb = load_object('gboost.dat')
logit = load_object('logit.dat')
rf = load_object('rf.dat')
sammer = load_object('sammer.dat')
svm = load_object('svm.dat')

gb_roc_dict = get_fpr_tpr(gb, x_corr, y_corr.iloc[:, 1])
logit_roc_dict = get_fpr_tpr(logit, x_corr, y_corr.iloc[:, 1])
rf_roc_dict = get_fpr_tpr(rf, x_corr, y_corr.iloc[:, 1])
sammer_roc_dict = get_fpr_tpr(sammer, x_corr, y_corr.iloc[:, 1])
svm_roc_dict = get_fpr_tpr(svm, x_corr, y_corr.iloc[:, 1])

clf_dict = {}
clf_dict['gb'] = gb_roc_dict
clf_dict['logit'] = logit_roc_dict
clf_dict['rf'] = rf_roc_dict
clf_dict['sammer'] = sammer_roc_dict
clf_dict['svm'] = svm_roc_dict
plot_ROC_curve_classes(clf_dict)
########################################
