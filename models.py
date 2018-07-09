
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

def score_test(model, x_test):
    from pandas import DataFrame
    y_pred = model.predict(x_test)
    labels = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    return DataFrame({
        'precision' : precision_score(y_test.attack_type, y_pred, average=None, labels=labels),
        'recall' : recall_score(y_test.attack_type, y_pred, average=None, labels=labels),
        'F1' : f1_score(y_test.attack_type, y_pred, average=None, labels=labels),
        'kappa' : cohen_kappa_score(y_test.attack_type, y_pred, labels=labels) },
        # 'AUC' : roc_auc_score(y_test.attack_type, y_pred, average=None) },
        index=labels)

######################################################################
#   L O G I S T I C

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

gb = GradientBoostingClassifier(max_depth=20, n_estimators=50, random_state=4129)
%time gb.fit(x_train_r, y_train_r)
score_test(gb, x_test)
