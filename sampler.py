# uses the imbalanced-learn library
# http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html
# install: conda install -c conda-forge imbalanced-learn

from dataload import *

######################################################################
#   R E S A M P L I N G

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler #, SMOTE, ADASYN
# from imblearn.combine import SMOTEENN, SMOTETomek

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