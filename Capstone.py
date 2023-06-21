from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


df = pd.read_csv('wwc_matches.csv')

outcome_data = df.copy()
le = LabelEncoder()
outcome_data["date"] = le.fit_transform(outcome_data["date"])
outcome_data["league"] = le.fit_transform(outcome_data["league"])
outcome_data["team1"] = le.fit_transform(outcome_data["team1"])
outcome_data["team2"] = le.fit_transform(outcome_data["team2"])
outcome_data["spi1"] = le.fit_transform(outcome_data["spi1"])
outcome_data["spi2"] = le.fit_transform(outcome_data["spi2"])
outcome_data["prob1"] = le.fit_transform(outcome_data["prob1"])
outcome_data["prob2"] = le.fit_transform(outcome_data["prob2"])
outcome_data["probtie"] = le.fit_transform(outcome_data["probtie"])
outcome_data["proj_score1"] = le.fit_transform(outcome_data["proj_score1"])
outcome_data["proj_score2"] = le.fit_transform(outcome_data["proj_score2"])
outcome_data["xg1"] = le.fit_transform(outcome_data["xg1"])
outcome_data["xg2"] = le.fit_transform(outcome_data["xg2"])
outcome_data["nsxg1"] = le.fit_transform(outcome_data["nsxg1"])
outcome_data["nsxg2"] = le.fit_transform(outcome_data["nsxg2"])
outcome_data["adj_score1"] = le.fit_transform(outcome_data["adj_score1"])
outcome_data["adj_score2"] = le.fit_transform(outcome_data["adj_score2"])
print(outcome_data["team1"], outcome_data["team2"])

outcome_data["outcome"] = np.where(outcome_data["adj_score1"] > outcome_data["adj_score2"], 1, 0)
print(outcome_data["outcome"])
x = outcome_data.drop("outcome", axis=1)
y = outcome_data["outcome"]

sc = StandardScaler()
x = sc.fit_transform(x)

num_folds = 10
seed = 42

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

print("Train and Test subsets size after splitting", np.shape(x_train), np.shape(x_test))

models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('ET', ExtraTreesClassifier()))
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

models.append(('AB', AdaBoostClassifier()))
ab = AdaBoostClassifier()

best_model = ab
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# ab.fit(x_train, y_train)
# predictions = ab.predict(x_test)
#
# print("Accuracy: ", accuracy_score(y_test, predictions))

# print(classification_report(y_test, y_pred))

best_model = ab
best_model.fit(x_train, y_train)
ab_roc_auc = roc_auc_score(y_test, best_model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:, 1])

plt.figure()
plt.plot(fpr, tpr, label='AdaBoostClassifier(area = %0.2f)' % ab_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('LOC_ROC')
plt.show()

# This part messing up rn so im leaving it for now:
# for x in range(len(y_pred)):
#     print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x])

# This part they wanted us to use logistic regression aswell
LR = LogisticRegression()
parameters = {'penalty': ('l1', 'l2', 'elasticnet', 'none'),
              'C': [1, 100, 500, 1000],
              'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}
model = GridSearchCV(LR, parameters).fit(x_train, y_train)
print(model.best_params_)

model = LogisticRegression(C=1, penalty='l2', solver='newton-cg',
                           max_iter=100, random_state=42).fit(x_train, y_train)

tr_score = model.score(x_train, y_train)
print("Training Set Score:", tr_score)

y_pred = model.predict(x_test)
tst_score = (model.score(x_test, y_test))
print("Test Set Score:", tst_score)

score = accuracy_score(y_test, y_pred)
print("Accuracy score: ", score)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

cl_report = classification_report(y_test, y_pred)
print(cl_report)

lr_roc_auc = roc_auc_score(y_test, model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression(area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('LOC_ROC')
plt.show()

matrix = classification_report(y_test, y_pred)
print(matrix)

for x in range(len(y_pred)):
    print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x])

import joblib

model_filename = 'best_model.sav'
joblib.dump(ab, model_filename)
# import joblib
#
# joblib.dump(best_model, "CapstoneC.sav")
