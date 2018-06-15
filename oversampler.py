# http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html
# install: conda install -c conda-forge imbalanced-learn

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

ros = RandomOverSampler(random_state=4129)
x_train_r, y_train_r = ros.fit_sample(x_train.iloc[:, 4:], y_train)
# ? it only takes numeric features ?

# other samplers to try
# x_train_r, y_train_r = SMOTE(random_state=4129).fit_sample(x_train, y_train)
# x_train_r, y_train_r = ADASYN(random_state=4129).fit_sample(x_train, y_train)
