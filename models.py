
# Counter(y_train.attack)
# Counter(y_train.attack_type)

from dataload import load_object, save_object
# x_train, y_train = load_object('train.dat')
x_train_r, y_train_r = load_object('train_r.dat')
x_test, y_test = load_object('test.dat')

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

def score_test(model):
    from pandas import DataFrame
    y_pred = model.predict(x_test.iloc[:, 4:])
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
logit = LogisticRegression(C=21, multi_class='ovr', solver='newton-cg', n_jobs=-1)
logit.fit(x_train_r, y_train_r)
score_test(logit)

# fit on imbalanced data
logit = LogisticRegression(C=31, multi_class='ovr', solver='sag', n_jobs=-1)
logit.fit(x_train.iloc[:, 4:], y_train.attack)
logit.score(x_test.iloc[:, 4:], y_test.attack)

######################################################################
#   S V M
