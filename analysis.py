# this code performs the analysis using functions in the other py files

%cd Documents/ML/Project
from dataload import *
x_train, y_train = load_object('train.dat')
x_test, y_test = load_object('test.dat')
cats_train, cats_test, catsvd_train, catsvd_test = load_object('cats.dat')
# x_train, x_test, y_train, y_test = load_object('ready.dat')

###########################################################
#   skip these if you're loading from .dat files

# load data from file, process it, then save as train/test/cats.dat files
# make_data()

###########################################################

#   FEATURE SCALING
# we need to minmaxscale x_train to [0,1].

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler(copy=False)      # do it in-place
mms.fit_transform(x_train)
mms.transform(x_test)
# catsvd_train/test these don't need scaling, they are binary dummy vars

# join catsvd_train/test as well as the 9 important raw cols of cats_train/test
# to x_train/test
# this also converts x_train/test into np.array format (ie we lose column names)
# now all columns are numeric.

# 56 cols: x_train cols 0-36, catsvd_train cols 37-46, cats_train cols 47-55

cols = [2,8,9,10,13,33,34,58,74]
x_train = np.hstack([x_train, catsvd_train, cats_train[:, cols].toarray()])
x_test = np.hstack([x_test, catsvd_test, cats_test[:, cols].toarray()])
del cats_train, cats_test, catsvd_train, catsvd_test

# at this point, x_train/x_test are ready for resampling/modelling
save_object([x_train, x_test, y_train, y_test], 'ready.dat')

######################################################################

# from collections import Counter
# Counter(y_train.attack[:120])
# Counter(y_train.attack_type[:1000])
# Counter(y_train.attack)
# Counter(y_train.attack_type)

#   RESAMPLING
# resampling is done immediately prior to model-fitting

# make a sampling pipeline on the full training set (target = attack)
# samp_pipe = make_pipe(19999, levels=['smurf', 'neptune', 'normal'])
# x_train_r, y_train_r = samp_pipe.fit_sample(x_train, y_train.attack)

# make a sampling pipeline on the full training set (target = attack_type)
samp_pipe = make_pipe(19999, levels=['dos', 'normal', 'probe'])
x_train_r, y_train_r = samp_pipe.fit_sample(x_train, y_train.attack_type)

# x_train_r, y_train_r now contain the training set with balanced classes.
# use them for modelling.


######################################################################
# Decision Tree for variable importance (one-hot encoded categoricals)

def cats_importance(cats_train, target):
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=10, random_state=4129)
    dt.fit(cats_train, y_train.attack_type == target)
    return pd.DataFrame(dt.feature_importances_, columns=['imp']).sort_values(by='imp', axis=0, ascending=False).iloc[:20]
# [cats_importance(cats_train, i) for i in ['normal','dos','probe','r2l','u2r']]

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
