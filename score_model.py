import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import DataDimensionalityWarning
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
# pd.set_option('display.max_columns', None)

df = pd.read_csv('wwc_matches.csv')
df["outcome"] = np.where(df["score1"] > df["score2"], 1, 0)

for col in df:
    if df[col].dtype == 'object':
        df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))

print(df)

# class_label = df["outcome"]
# df = df.drop(["outcome"], axis=1)
# df = (df - df.min())/(df.max() - df.min())
# df["outcome"] = class_label
# print(df)

outcome_data = df.copy()
le = preprocessing.LabelEncoder()
date = le.fit_transform(list(outcome_data["date"]))
league_id = le.fit_transform(list(outcome_data["league_id"]))
league = le.fit_transform(list(outcome_data["league"]))
team1 = le.fit_transform(list(outcome_data["team1"]))
team2 = le.fit_transform(list(outcome_data["team2"]))
spi1 = le.fit_transform(list(outcome_data["spi1"]))
spi2 = le.fit_transform(list(outcome_data["spi2"]))
prob1 = le.fit_transform(list(outcome_data["prob1"]))
prob2 = le.fit_transform(list(outcome_data["prob2"]))
probtie = le.fit_transform(list(outcome_data["probtie"]))
proj_score1 = le.fit_transform(list(outcome_data["proj_score1"]))
proj_score2 = le.fit_transform(list(outcome_data["proj_score2"]))
score1 = le.fit_transform(list(outcome_data["score1"]))
score2 = le.fit_transform(list(outcome_data["score2"]))
xg1 = le.fit_transform(list(outcome_data["xg1"]))
xg2 = le.fit_transform(list(outcome_data["xg2"]))
nsxg1 = le.fit_transform(list(outcome_data["nsxg1"]))
nsxg2 = le.fit_transform(list(outcome_data["nsxg2"]))
adj_score1 = le.fit_transform(list(outcome_data["adj_score1"]))
adj_score2 = le.fit_transform(list(outcome_data["adj_score2"]))
outcome = le.fit_transform(list(outcome_data["outcome"]))

x = list(zip(team1, team2))
y = outcome

num_folds = 5
seed = 42
scoring = 'accuracy'

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20, random_state=seed)

print(np.shape(x_train), np.shape(x_test))

models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
results = []
names = []

print("Performance on Training set")

for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    msg += '\n'
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()

best_model = rf
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

best_model = rf
best_model.fit(x_train, y_train)
rf_roc_auc = roc_auc_score(y_test, best_model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:, 1])

plt.figure()
plt.plot(fpr, tpr, label='Random Forest(area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('LOC_ROC')
plt.show()

for x in range(len(y_pred)):
    print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)

LR = LogisticRegression()
parameters = {'penalty': ('l1', 'l2', 'elasticnet', 'none'),
              'C': [1, 100, 500, 1000],
              'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}
model = GridSearchCV(LR, parameters).fit(x_train,y_train)
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
    print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)
