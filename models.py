
# Counter(y_train.attack)
# Counter(y_train.attack_type)

from dataload import load_object, save_object
# x_train, y_train = load_object('train.dat')
x_train_r, y_train_r = load_object('train_r.dat')
x_test, y_test = load_object('test.dat')

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

######################################################################
#   L O G I S T I C

# fit on balanced data
logit = LogisticRegression(C=21, multi_class='ovr', solver='newton-cg', n_jobs=-1)
logit.fit(x_train_r, y_train_r)

def score_test():
    from pandas import DataFrame
    # logit.score(x_test.iloc[:, 4:], y_test.attack)
    y_pred = logit.predict(x_test.iloc[:, 4:])
    return DataFrame({
        'accuracy' : [accuracy_score(y_test.attack_type, y_pred)],
        'recall' : [recall_score(y_test.attack_type, y_pred, average='micro')],
        'F1' : [f1_score(y_test.attack_type, y_pred, average='micro')] })

# fit on imbalanced data
logit = LogisticRegression(C=31, multi_class='ovr', solver='sag', n_jobs=-1)
logit.fit(x_train.iloc[:, 4:], y_train.attack)
logit.score(x_test.iloc[:, 4:], y_test.attack)

score_test()

######################################################################
#   S V M
