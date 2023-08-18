import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('wwc_matches.csv')

outcome_data = df.copy()
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        outcome_data[col] = le.fit_transform(df[col])

team_le = LabelEncoder()
team_le.fit(pd.concat([df["team1"], df["team2"]]).unique())

outcome_data["outcome"] = np.where(df["score1"] > df["score2"], 0, 1)

features = ['team1', 'team2']

x = outcome_data[features]
y = outcome_data['outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

best_model = RandomForestClassifier()
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
model_accuracy = accuracy_score(y_test, y_pred)


def predict_winner(team1, team2):
    team1_encoded = team_le.transform([team1])
    team2_encoded = team_le.transform([team2])

    input_data = pd.DataFrame({'team1': [team1_encoded[0]], 'team2': [team2_encoded[0]]})

    prediction = best_model.predict(input_data)
    return 'Team 1' if prediction == 0 else 'Team 2'

# References:
# Preprocessing Variables:
# Preprocessing df with sklearn - https://scikit-learn.org/stable/modules/preprocessing.html
# df Preprocessing with Python - https://towardsdfscience.com/df-preprocessing-with-python-pandas-part-1-missing-df-45e76b781993

# Feature and Target Variables:
# Features and target variable in Machine Learning - https://towardsdfscience.com/understanding-features-vs-target-in-machine-learning-df49baf20b43

# RandomForestClassifier Parameters:
# Random Forest Classifier in sklearn - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Understanding Random Forest - https://towardsdfscience.com/understanding-random-forest-58381e0602d2

# np.select:
# numpy.select documentation - https://numpy.org/doc/stable/reference/generated/numpy.select.html
# Using np.select for if-elif-else conditions - https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dfframe-column

# Preprocessing Together:
# Feature Scaling with sklearn - https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
# Encoding categorical variables with sklearn - https://scikit-learn.org/stable/modules/preprocessing_targets.html#preprocessing-targets