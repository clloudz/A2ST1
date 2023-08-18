import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("wwc_matches.csv")

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.nunique())
print(df.info)
print(df.describe())

# Visualising data
fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
df.hist(ax=ax, bins=30)
plt.show()

# Detecting Outliers / Drop outliers
df.plot(kind='box', subplots=True,
        layout=(4, 4), sharex=False, sharey=False, figsize=(20, 10), color='deeppink')
plt.show()

continuous_variables = ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2',
                                'score1', 'score2', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2']


def outliers(df_out, drop=False):
    for each_variable in df_out.columns:
        variable_data = df_out[each_variable]
        q1 = np.percentile(variable_data, 25.)
        q3 = np.percentile(variable_data, 75.)
        iqr = q3 - q1
        outlier_step = iqr * 1.5
        outliers = variable_data[~((variable_data >= q1 -
                                    outlier_step) & (variable_data <= q3 + outlier_step))].index.tolist()
        if not drop:
            print('For the variable {}, No of Outliers is {}'.format(each_variable, len(outliers)))

outliers(df[continuous_variables], drop=False)

continuous_variables = ['spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2',
                        'score1', 'score2', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2']


def outliers(df_out, drop=False):
    for each_variable in df_out.columns:
        variable_data = df_out[each_variable]
        q1 = np.percentile(variable_data, 25.)
        q3 = np.percentile(variable_data, 75.)
        iqr = q3-q1
        outlier_step = iqr * 1.5
        outliers = variable_data[~((variable_data >= q1 -
                                    outlier_step) & (variable_data <= q3 + outlier_step))].index.tolist()
        if not drop:
            print('For the variable {}, No of Outliers is {}'.format(each_variable, len(outliers)))
        if drop:
            df.drop(outliers, inplace=True, errors='ignore')
            print('Outliers from {} variable removed'.format(each_variable))


outliers(df[continuous_variables], drop=True)


df.plot(kind='box', subplots=True,
        layout=(4, 4), sharex=False, sharey=False,
        figsize=(20, 10), color='deeppink')
plt.show()

print(df.shape)

# Value Distribution
print(df.score1.value_counts())
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

plt.figure()

score = df.score1.value_counts()
p1 = plt.bar(np.arange(len(score)), score)

for perc in p1:
    height = perc.get_height()
    plt.annotate("{}%".format(height), (perc.get_x() + perc.get_width()/2, height+.05), ha="center", va="bottom",
                 fontsize=15)

plt.show()


fig, ax = plt.subplots(layout="constrained")
ax = df.score1.value_counts().plot(kind='bar')
ax.set_title("Match Score Frequency", fontsize=13, weight='bold')

totals = []
for i in ax.patches:
    totals.append(i.get_height())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_x()+0.09, i.get_height() - 50, str(round((i.get_height()/total)*100, 2))+'%', fontsize=14,
            color='white', weight='bold')
plt.show()

numeric_columns = df.select_dtypes(include=['int', 'float']).columns
df_numeric = df[numeric_columns]

# Set plot style and figure size
sns.set(style="white")
plt.rcParams['figure.figsize'] = (15, 10)

# Create correlation heatmap with numeric columns
sns.heatmap(df_numeric.corr(), annot=True, linewidths=.5, cmap="Blues")
plt.title('Correlation Between Variables', fontsize=30)
plt.show()

profile = ProfileReport(df,title="Heart Disease EDA",
                        html={'style' : {'full_width' : True}})
