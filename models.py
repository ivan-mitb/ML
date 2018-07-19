from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score

def score_test(model, x_test, y_test):
    from pandas import DataFrame
    y_pred = model.predict(x_test)
    # recode NaNs in y_test.attack_type
    y_recode = y_test.attack_type
    y_recode[y_recode.isna()] = 'unknown'
    labels = ['normal', 'dos', 'probe', 'r2l', 'u2r', 'unknown']
    return DataFrame({
        'precision' : precision_score(y_recode, y_pred, average=None, labels=labels),
        'recall' : recall_score(y_recode, y_pred, average=None, labels=labels),
        'F1' : f1_score(y_recode, y_pred, average=None, labels=labels) },
        # 'kappa' : cohen_kappa_score(y_test.attack_type, y_pred, labels=labels) },
        # 'AUC' : roc_auc_score(y_test.attack_type, y_pred, average=None) },
        index=labels)

# normal-vs-rest recoding of classes
def score_test2(model, x_test, y_test):
    from pandas import DataFrame
    y_pred = model.predict(x_test)
    # recode non-normal classes in y_test.attack_type
    y_recode = y_test.attack_type
    y_recode[y_recode.isna()] = 'unknown'
    y_recode[y_recode != 'normal'] = 'attack'
    y_pred[y_pred != 'normal'] = 'attack'
    labels = ['normal', 'attack']
    return DataFrame({
        'precision' : precision_score(y_recode, y_pred, average=None, labels=labels),
        'recall' : recall_score(y_recode, y_pred, average=None, labels=labels),
        'F1' : f1_score(y_recode, y_pred, average=None, labels=labels) },
        # 'AUC' : roc_auc_score(y_recode, y_pred, average=None) },
        index=labels)

######################################################################
#   L O G I S T I C

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# fit on balanced data
logit = LogisticRegressionCV(Cs=[100,500,750,900], scoring='recall', multi_class='ovr', solver='saga', random_state=4129, n_jobs=-1)
%time logit.fit(x_train_r, y_train_r)
logit.C_
score_test(logit)

# fit on imbalanced data
logit = LogisticRegression(C=31, multi_class='ovr', solver='saga', n_jobs=-1, random_state=4129)
%time logit.fit(x_train_r, y_train_r)
# logit.score(x_test, y_test.iloc[:,1])
score_test(logit)
%time logit.fit(x_train[:, 18:37], y_train)
logit.score(x_test[:, :37], y_test[:,1])

######################################################################
#   S V M

from sklearn.svm import LinearSVC, SVC

svc = LinearSVC(C=1, random_state=4129)
%time svc.fit(x_train_r[:, 18:37], y_train_r)
score_test(svc, x_test[:, 18:37])
%time svc.fit(x_train_r, y_train_r)
score_test(svc, x_test)

# kernel SVM
svc = SVC(C=1, random_state=4129)
%time svc.fit(x_train_r[:, 18:37], y_train_r)
score_test(svc)
%time svc.fit(x_train_r, y_train_r)
score_test(svc)

######################################################################
#   GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(max_depth=20, n_estimators=50, verbose=1, random_state=4129)
%time gb.fit(x_train_r, y_train_r)
score_test(gb, x_test, y_test)
score_test(gb, x_corr, y_corr)
score_test2(gb, x_corr, y_corr)

# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
# https://datascience.stackexchange.com/questions/14377/tuning-gradient-boosted-classifiers-hyperparametrs-and-balancing-it

from sklearn.model_selection import GridSearchCV, StratifiedKFold
parameters = {'max_depth':[20,50], 'n_estimators':[100,150,250,500]}
cv = StratifiedKFold(3, shuffle=True, random_state=4129)
gs = GridSearchCV(gb, parameters, n_jobs=1, cv=cv, verbose=1, return_train_score=False)
%time gs.fit(x_train_r, y_train_r)
gs.best_params_
score_test(gs.best_estimator_, x_test, y_test)
gs.best_score_
gs.cv_results_


######################################################################
#   Random Forest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

param_rand = {"criterion" : ['gini', 'entropy'],
                        "n_estimators": randint(10, 50),
                        "max_depth": randint(15, 25),
                        "max_features": randint(5, 15),
                        "min_samples_split": randint(2, 11),
                        "min_samples_leaf": randint(1, 11),
                        "bootstrap": [True, False]}

param_rand2 = {"criterion" : ['gini', 'entropy'],
                        "n_estimators": randint(25, 66),
                        "max_depth": randint(15, 25),
                        "max_features": randint(5, 15),
                        "min_samples_split": randint(2, 11),
                        "min_samples_leaf": randint(1, 11),
                        "bootstrap": [True, False]}

skf_shuffle = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 4129)
clf_est = RandomForestClassifier(n_jobs = -1, random_state=4129)

random_search = RandomizedSearchCV(estimator = clf_est, param_distributions = param_rand2, 
#                                   cv = skf_shuffle, 
                                   n_jobs = -1)
random_search.fit(x_train_r, y_train_r)
print('best estimator:', random_search.best_estimator_)
print('best parameters:', random_search.best_params_)
print('best score:', random_search.best_score_)
score_test(random_search, x_test)

param_grid = {"criterion" : ['gini', 'entropy'],
                    "n_estimators": [42, 44, 46],
                    "max_features": [8, 9, 10],
                    "max_depth": [20, 21, 22],
                    "min_samples_split": [2, 3, 4],
                    "min_samples_leaf": [1, 2, 3],
                    "bootstrap": [True, False]}

random_search = GridSearchCV(estimator = clf_est, param_grid = param_grid, 
#                                   cv = skf_shuffle, 
                                   n_jobs = -1)
random_search.fit(x_train_r, y_train_r)
print('best estimator:', random_search.best_estimator_)
print('best parameters:', random_search.best_params_)
print('best score:', random_search.best_score_)
score_test(random_search, x_test)

im
param_grid_cfm = {'max_depth': [18,19,20], 
                  'max_features': [8,9,10], 
                  'min_samples_leaf': [1,2], 
                  'min_samples_split': [8,9,10], 
                  'n_estimators': [50,51,52]}
clf_cfm = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='entropy',n_jobs = -1, random_state=4129)
random_search = GridSearchCV(estimator = clf_cfm, param_grid = param_grid_cfm,                           cv = skf_shuffle, n_jobs = -1)
random_search.fit(x_train_r, y_train_r)
print('best estimator:', random_search.best_estimator_)
print('best parameters:', random_search.best_params_)
print('best score:', random_search.best_score_)
score_test(random_search, x_test)

best_rf = RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=19, max_features=9,
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=9, min_weight_fraction_leaf=0.0,
            n_estimators=51, n_jobs=-1, oob_score=False, random_state=4129,
            verbose=0, warm_start=False)
best_rf.fit(x_train_r, y_train_r)
score_test(best_rf, x_test, y_test)
'''
         precision    recall        F1
normal    0.999825  0.998664  0.999244
dos       0.999997  0.999807  0.999902
probe     0.962963  0.999513  0.980898
r2l       0.788732  0.991150  0.878431
u2r       0.714286  1.000000  0.833333
unknown   0.000000  0.000000  0.000000
'''
x_corr, y_corr = load_object('correduce.dat')
score_test(best_rf, x_corr, y_corr)
'''
         precision    recall        F1
normal    0.733688  0.993729  0.844135
dos       0.997986  0.998616  0.998301
probe     0.506178  0.930585  0.655699
r2l       0.895118  0.165193  0.278913
u2r       0.311111  0.358974  0.333333
unknown   0.000000  0.000000  0.000000
'''
score_test2(best_rf, x_corr, y_corr)
'''
        precision    recall        F1
normal   0.733688  0.993729  0.844135
attack   0.998340  0.912728  0.953617
'''

best_grid_rf = RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=19, max_features=8,
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=9, min_weight_fraction_leaf=0.0,
            n_estimators=52, n_jobs=-1, oob_score=False, random_state=4129,
            verbose=0, warm_start=False)
best_grid_rf.fit(x_train_r, y_train_r)
score_test(best_grid_rf, x_test, y_test)
'''
         precision    recall        F1
normal    0.999784  0.998571  0.999177
dos       0.999997  0.999791  0.999894
probe     0.963875  0.999757  0.981488
r2l       0.725490  0.982301  0.834586
u2r       0.500000  1.000000  0.666667
'''
#x_corr, y_corr = load_object('correduce.dat')
score_test(best_grid_rf, x_corr, y_corr)
'''
         precision    recall        F1
normal    0.734917  0.993877  0.845002
dos       0.998262  0.998303  0.998283
probe     0.495526  0.931847  0.646999
r2l       0.810277  0.171033  0.282447
u2r       0.372093  0.410256  0.390244
unknown   0.000000  0.000000  0.000000
'''
score_test2(best_grid_rf, x_corr, y_corr)
'''        precision    recall        F1
normal   0.734917  0.993877  0.845002
attack   0.998381  0.913263  0.953927'''


