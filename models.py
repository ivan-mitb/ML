
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
