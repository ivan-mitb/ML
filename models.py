from dataload import load_object, save_object

# Counter(y_train.attack)
# Counter(y_train.attack_type)

# x_train, y_train = load_object('train.dat')
x_test, y_test = load_object('test.dat')

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
logit = LogisticRegression(C=31, multi_class='ovr', solver='newton-cg', n_jobs=-1)
logit.fit(x_train_r, y_train_r)
logit.score(x_test.iloc[:, 4:], y_test.attack)

y_pred = logit.predict(x_test.iloc[:, 4:])
accuracy_score(y_test.attack, y_pred)
recall_score(y_test.attack, y_pred, average='micro')
f1_score(y_test.attack, y_pred, average='micro')

logit = LogisticRegression(C=31, multi_class='ovr', solver='sag', n_jobs=-1)
logit.fit(x_train.iloc[:, 4:], y_train.attack)
logit.score(x_test.iloc[:, 4:], y_test.attack)

y_pred = logit.predict(x_test.iloc[:, 4:])
accuracy_score(y_test.attack, y_pred)
recall_score(y_test.attack, y_pred, average='micro')
f1_score(y_test.attack, y_pred, average='micro')
