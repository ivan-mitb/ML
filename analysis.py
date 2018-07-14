# this code performs the analysis using functions in the other py files

%cd Documents/ML/Project
from dataload import *

###########################################################

# READY.DAT - 56 cols
# 56 cols: KDD cols 0-36, catsvd_train cols 37-46, cats_train cols 47-55
# 37 KDD cols + 10 tSVD + 9 bestcats
# generated by make_data()

x_train, x_test, y_train, y_test = load_object('ready.dat')
x_train.max(axis=0)
# this is the expected output:
# array([1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.72155493, 1.60947741, 1.18151578,
#        1.16779543, 0.98840791, 1.09742489, 1.13992441, 0.81566507,
#        1.01021012, 1.2157684 , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        ])

# REDUCE.DAT - 35 cols
# reduce the 37 KDD columns into 5 princomps + 11 best, + 10 tSVD + 9 bestcats
# generated by make_reduce(x_train, x_test, y_train, y_test)

x_train, x_test, y_train, y_test = load_object('reduce.dat')
x_train.max(axis=0)
# array([2.27264385, 1.86312597, 2.20028144, 1.53789152, 1.65899287,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.72155493, 1.60947741, 1.18151578, 1.16779543,
#        0.98840791, 1.09742489, 1.13992441, 0.81566507, 1.01021012,
#        1.2157684 , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ])

# CORRECTED.DAT - 56 cols
# the KDD test set
x_corr, y_corr = load_object('corrected.dat')
x_corr.shape, y_corr.shape
x_corr.max(axis=0)

######################################################################
#   RESAMPLING

# resampling is done immediately prior to model-fitting

from sampler import make_pipe

# make a sampling pipeline on the full training set (target = attack)
# samp_pipe = make_pipe(19999, levels=['smurf', 'neptune', 'normal'])
# x_train_r, y_train_r = samp_pipe.fit_sample(x_train, y_train.attack)

# make a sampling pipeline on the full training set (target = attack_type)
samp_pipe = make_pipe(29999, levels=['dos', 'normal', 'probe'])
x_train_r, y_train_r = samp_pipe.fit_sample(x_train, y_train.attack_type)

# x_train_r, y_train_r now contain the training set with balanced classes.
# use them for modelling.

######################################################################
#   MODELLING

# run the code interactively in models.py



######################################################################
# Decision Tree for variable importance (one-hot encoded categoricals)

def cats_importance(cats_train, target):
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=10, random_state=4129)
    dt.fit(cats_train, y_train.attack_type == target)
    return pd.DataFrame(dt.feature_importances_, columns=['imp']).sort_values(by='imp', axis=0, ascending=False).iloc[:20]
# [cats_importance(cats_train, i) for i in ['normal','dos','probe','r2l','u2r']]

#   more feature selection and PCA on cols 0:37
#   Nic

def nic_col_select():
    # get list of column that accounts for variance of minority attack types
    r2l_impts = cats_importance(x_train[:,0:37], 'r2l')
    u2r_impts = cats_importance(x_train[:,0:37], 'u2r')
    t_percent_var = 0.0
    col_list = []
    for i in range(20):
        col_list.append(r2l_impts.index[i])
        t_percent_var += r2l_impts.iloc[i, 0]
        if(t_percent_var > 0.85):
            break
    t_percent_var = 0.0
    for i in range(20):
        if(u2r_impts.index[i] not in col_list):
            col_list.append(u2r_impts.index[i])
    #    else:
    #        print('duplicate')
        t_percent_var += u2r_impts.iloc[i, 0]
        if(t_percent_var > 0.85):
            break
    # print(sorted(col_list))

    ## Get top 5 PCA components
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 5, random_state = 4129)
    pca_result = pca.fit_transform(x_train[:, :37])
    # pca.explained_variance_ratio_

    # join the 5 princomps with the 11 best columns
    return np.hstack((pca_result, x_train[:, col_list])), np.hstack((pca.transform(x_test[:, :37]), x_test[:, col_list]))
# x_train_reduce, x_test_reduce = nic_col_select()

######################################################################
# the key question here is: are there features which are highly correlated with the rare classes?
# we should use such features directly in the classifiers -- don't pass them thru PCA.

'''
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

dt = DecisionTreeClassifier(max_depth=10, random_state=4129)
%time dt.fit(x_train, y_train.attack_type)
%time dt.fit(x_train, y_train.attack_type == 'normal')    # fit on rare class r2l, u2r
pd.DataFrame(dt.feature_importances_, index=x_train.columns, columns=['imp']).sort_values(by='imp', axis=0, ascending=False).iloc[:16]


# GridSearch for params
parameters = [
    {'criterion': ['gini','entropy'], 'splitter': ['best','random'],
    'max_depth' : range(10,25)}]
cv = StratifiedKFold(3, shuffle=True, random_state=4129)
gs = GridSearchCV(dt, parameters, n_jobs=-1, cv=cv, return_train_score=True)
%time gs.fit(x_train, y_train)
'''

######################################################################

# nick's feature_transformation.py

'''
x_train, y_train = load_object('train.dat')

x_train_dum = cat2dummy(x_train)
scale_list = scale_list(x_train_dum)

x_train_mms = minmax(x_train_dum, scale_list)

##print to compare min-max before and after scaling
for i in range(x_train_dum.shape[1]):
    print(i, " : Scaled>> ", min(x_train_mms.iloc[:, i]), '-', max(x_train_mms.iloc[:, i]),
          ', UnScaled>>', min(x_train_dum.iloc[:, i]), '-', max(x_train_dum.iloc[:, i]))

x_train_bin = binning(x_train_mms, scale_list)

### Save to object
save_object([x_train_bin, y_train], 'train_bin.dat')
### Save to view
# np.savetxt("x_train_bin.csv", x_train_bin, delimiter=",")


## UNUSED
x_train_rs = robust(x_train_dum, scale_list)
np.savetxt("x_train_rs.csv", x_train_rs, delimiter=",")
x_train_qt = quantile(x_train_dum, scale_list)
np.savetxt("x_train_qt.csv", x_train_qt, delimiter=",")
x_train_logtf = log_transform(x_train_dum, scale_list)
np.savetxt("x_train_logtf.csv", x_train_logtf, delimiter=",")
'''
