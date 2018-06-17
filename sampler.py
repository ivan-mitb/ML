# uses the imbalanced-learn library
# http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html
# install: conda install -c conda-forge imbalanced-learn

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler #, SMOTE, ADASYN
# from imblearn.combine import SMOTEENN, SMOTETomek

from dataload import load_object, save_object
# x_train, y_train = load_object('train.dat')
# x_test, y_test = load_object('test.dat')

# ? it only takes numeric features ?
# categoricals have to be transformed using one-hot encoding

# other samplers to try
# x_train_r, y_train_r = SMOTE(random_state=4129).fit_sample(x_train, y_train)
# x_train_r, y_train_r = ADASYN(random_state=4129).fit_sample(x_train, y_train)

def make_pipe(kcount=100000, levels=['smurf', 'neptune', 'normal']):
    '''Make a pipeline of an under- and an over-sampler, that produces
    kcount samples for each target classself.
    levels: a list of target classes to under-sample.'''

    ratio = dict(zip(levels, [kcount] * len(levels)))
    # ratio = {'smurf':kcount, 'neptune':kcount, 'normal':kcount}

    # down-sample majority classes to kcount
    rus = RandomUnderSampler(random_state=4129, ratio=ratio)

    # up-sample the others to kcount
    ros = RandomOverSampler(random_state=4129)

    from imblearn.pipeline import Pipeline
    return Pipeline([('under',rus), ('over',ros)])

# Counter(y_train.attack[:120])
# Counter(y_train.attack_type[:1000])
# Counter(y_train.attack)
# Counter(y_train.attack_type)

# make a sampling pipeline on the full training set (target = attack)
samp_pipe = make_pipe(20000, levels=['smurf', 'neptune', 'normal'])
x_train_r, y_train_r = samp_pipe.fit_sample(x_train.iloc[:, 4:], y_train.attack)

# make a sampling pipeline on the full training set (target = attack_type)
samp_pipe = make_pipe(20000, levels=['dos', 'normal', 'probe'])
x_train_r, y_train_r = samp_pipe.fit_sample(x_train.iloc[:, 4:], y_train.attack_type)

# x_train_r, y_train_r now contain the training set with balanced classes.
# use them for modelling.
