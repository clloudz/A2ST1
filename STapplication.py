import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import plotly.express as px
from prediction_model import predict_winner
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="2019 FIFA Women's World Cup", layout="wide",
                   initial_sidebar_state="expanded")

df = pd.read_csv('wwc_matches.csv')
df = df.dropna()

# Calculating winner of match
df['Winner'] = np.where(df['score1'] > df['score2'], df['team1'], df['team2'])

# Calculating Team 1 and 2 correlation coefficient
correlation_coefficient_team1, _ = pearsonr(df['spi1'], df['score1'])
correlation_text_team1 = f'Correlation Coefficient: {correlation_coefficient_team1:.2f}'
correlation_coefficient_team2, _ = pearsonr(df['spi2'], df['score2'])
correlation_text_team2 = f'Correlation Coefficient: {correlation_coefficient_team2:.2f}'

# Country from column 'team1' and 'team2'
df['country1'] = df['team1'].str.split().str[0]
df['country2'] = df['team2'].str.split().str[0]

# exception handling
exceptions = ['South Korea', 'South Africa', 'New Zealand']

# Create data dictionary that shows country and corresponding continent
country_to_continent = {
    'France': 'Europe',
    'Germany': 'Europe',
    'Spain': 'Europe',
    'Norway': 'Europe',
    'Australia': 'Oceania',
    'South Korea': 'Asia',
    'China': 'Asia',
    'South Africa': 'Africa',
    'Nigeria': 'Africa',
    'Italy': 'Europe',
    'Japan': 'Asia',
    'USA': 'North America',
    'England': 'Europe',
    'Netherlands': 'Europe',
    'Sweden': 'Europe',
    'Thailand': 'Asia',
    'Cameroon': 'Africa',
    'Scotland': 'Europe',
    'Jamaica': 'Africa',
    'Canada': 'North America',
    'Argentina': 'South America',
    'Chile': 'South America',
    'New Zealand': 'Oceania',
    'Brazil': 'South America'
}

# Getting continent from country by using data dictionary
df['continent1'] = df['country1'].map(country_to_continent)
df['continent2'] = df['country2'].map(country_to_continent)

# wins per continent for teams 1 and teams 2
win_counts_continent1 = df[df['score1'] > df['score2']].groupby('continent1').size().reset_index(name='wins')
win_counts_continent2 = df[df['score2'] > df['score1']].groupby('continent2').size().reset_index(name='wins')

# average probability of team winning
team1_prob = df.groupby('team1')['prob1'].mean()
team2_prob = df.groupby('team2')['prob2'].mean()

# target values
df["outcome"] = np.where(df["score1"] > df["score2"], 0, 1)

st.sidebar.title('Dataset Analysis')
home = st.sidebar.button('Home')
with st.sidebar.expander("Exploratory Data Analysis"):
    eda = st.selectbox('Initial EDA Analysis', ['Select', 'First and Last 5 rows of dataset',
                                                'Attributes, Values and Dataset Shape',
                                                'Dataset Information and Statistics',
                                                'Data Visualisation', 'Removing Outliers',
                                                'Distribution of Values',
                                                'Variable Correlation', 'Dataset Profile Report'])
    questions = st.selectbox('EDA Questions and Visualisation',
                             ['Select', '1. Is there a correlation between SPI and actual scores?',
                              '2. What was the distribution of wins per continent?',
                              '3. Were the projected scores accurate?',
                              '4. What was the probability of each team winning the World Cup?',
                              '5. Was the score distribution equal?'])


def run():
    st.title("Match Winner Prediction")

    team1 = st.selectbox('Select Team 1', df['team1'].unique())
    team2 = st.selectbox('Select Team 2', df['team2'].unique())

    if st.button("Predict"):
        winner = predict_winner(team1, team2)
        st.write('The predicted winner is: ' + winner)


if __name__ == '__main__':
    run()

if home:
    st.title('2019 FIFA Womens World Cup')
    st.write("This task uses the 2019 Womens World Cup Predictions to perform Exploratory Data Analysis and create "
             "a prediction model. It includes information on the odds of each side winning each match in the 2019 "
             "FIFA Womens World Cup as well as the odds of a draw. The dataset contains elements like the "
             " forecasts timestamp, the two teams involved in a match, and the tournaments stage. A model that "
             "considered elements including team strength, recent performance, and the venue of the "
             "match produced the anticipated victory probability. The dataset contains the estimated victory "
             "probabilities for both teams as well as the tie probability.", df)

if questions == '1. Is there a correlation between SPI and actual scores?':
    st.title('2019 Womens World Cup')
    st.header('Question 1. Is there a correlation between SPI and actual scores?')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Team 1 SPI VS Match Score')
        fig, ax = plt.subplots()
        ax.scatter(data=df, x='spi1', y='score1')  # plot data in scatter-plot
        ax.axhline(df['score1'].mean(), color='r', linestyle='--',
                   label='Mean Match Score')  # show mean score against score distribution
        ax.axvline(df['spi1'].mean(), color='b', linestyle='--',
                   label='Mean SPI Rating')  # show mean spi against spi values
        ax.axline((df['spi1'].mean(), df['score1'].mean()), slope=correlation_coefficient_team1, color='g',
                  linestyle='--', label='Correlation')  # show correlation coefficient for values
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader('Team 2 SPI VS Match Score')
        fig, ax = plt.subplots()
        ax.scatter(data=df, x='spi2', y='score1')
        ax.axhline(df['score2'].mean(), color='r', linestyle='--', label='Mean Match Score')
        ax.axvline(df['spi2'].mean(), color='b', linestyle='--', label='Mean SPI Rating')
        ax.axline((df['spi2'].mean(), df['score2'].mean()), slope=correlation_coefficient_team2, color='g',
                  linestyle='--', label='Correlation')
        ax.legend()
        st.pyplot(fig)

if questions == '2. What was the distribution of wins per continent?':
    st.title('2019 Womens World Cup')
    st.header('Question 2. What was the distribution of wins per continent?')
    plt.figure(figsize=(8, 6))
    plt.pie(win_counts_continent1['wins'], labels=win_counts_continent1['continent1'], autopct='%1.1f%%')
    plt.title('Distribution of Wins per Continent Team 1')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    plt.pie(win_counts_continent2['wins'], labels=win_counts_continent2['continent2'], autopct='%1.1f%%')
    plt.title('Distribution of Wins per Continent Team 2')
    st.pyplot(plt)

if questions == '3. Were the projected scores accurate?':
    st.title('2019 Womens World Cup')
    st.header('Question 3. Were the projected scores accurate?')
    # Graph 1
    fig, ax = plt.subplots(figsize=(18, 6))

    x_labels = df['team1']
    x_positions = list(range(len(x_labels)))

    ax.bar(x_positions, df['proj_score1'], width=0.4, label='Projected Score')
    ax.bar([pos + 0.4 for pos in x_positions], df['score1'], width=0.4, label='Score')

    ax.set_xticks([pos + 0.2 for pos in x_positions])
    ax.set_xticklabels(x_labels, rotation=90)

    ax.set_xlabel('Team 1')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Projected Scores and Match Scores')

    ax.legend()

    st.pyplot(fig)

    # Graph 2
    fig, ax = plt.subplots(figsize=(18, 6))

    x_labels = df['team2']
    x_positions = list(range(len(x_labels)))

    ax.bar(x_positions, df['proj_score2'], width=0.4, label='Projected score')
    ax.bar([pos + 0.4 for pos in x_positions], df['score2'], width=0.4, label='Score')

    ax.set_xticks([pos + 0.2 for pos in x_positions])
    ax.set_xticklabels(x_labels, rotation=90)

    ax.set_xlabel('Team 2')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Projected Scores and Match Scores')

    ax.legend()

    st.pyplot(fig)

if questions == '4. What was the probability of each team winning the World Cup?':
    st.title('2019 Womens World Cup')
    st.header('Question 4. What was the probability of each team winning the world cup?')
    fig, ax = plt.subplots()
    team1_prob.plot(kind='bar', ax=ax)
    ax.set_xlabel('Team 1')
    ax.set_ylabel('Probability')
    ax.set_title('Probability of Team 1 Winning the World Cup')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    team2_prob.plot(kind='bar', ax=ax)
    ax.set_xlabel('Team 2')
    ax.set_ylabel('Probability')
    ax.set_title('Probability of Team 2 Winning the World Cup')
    st.pyplot(fig)


if questions == '5. Was the score distribution equal?':
    st.title('2019 Womens World Cup')
    st.header('Question 5. Was the score distribution equal?')
    st.subheader('Distribution of Scores - Team 1')

    st.subheader('Box Plot - Team 1')
    fig = px.box(df, y='score1', orientation='h')
    st.plotly_chart(fig)

    st.subheader('Box Plot - Team 2')
    fig = px.box(df, y='score2', orientation='h')
    st.plotly_chart(fig)

if eda == 'First and Last 5 rows of dataset':
    st.title('2019 Womens World Cup')
    st.header('First and Last 5 rows of dataset')
    st.write(df.head(), df.tail())

if eda == 'Attributes, Values and Dataset Shape':
    st.title('2019 Womens World Cup')
    st.header('Attributes, Values and Dataset Shape')
    at1, at2, at3 = st.columns(3)
    with at1:
        st.write('The dataset attributes are:', df.columns)
    with at2:
        st.write('The values for each attribute are:', df.nunique())
    with at3:
        st.write('The shape of the data is:', df.shape)

if eda == 'Dataset Information and Statistics':
    st.title('2019 Womens World Cup')
    st.header('Dataset Information and Statistics')
    st.write(df.info, df.describe())

if eda == 'Data Visualisation':
    st.title('2019 Womens World Cup')
    st.header('Data Visualisation')
    fig, ax = plt.subplots(figsize=(20, 20))
    df.hist(ax=ax, bins=30)
    st.pyplot(fig)

if eda == 'Removing Outliers':
    st.title('2019 Womens World Cup')
    st.header('Removing Outliers')
    checkbox_oj = st.checkbox('Original Data')

    if checkbox_oj:
        fig, axes = plt.subplots(figsize=(20, 20))
        df.plot(kind='box', subplots=True, layout=(5, 5), sharex=False, sharey=False, ax=axes, color='deeppink')
        st.pyplot(fig)

    checkbox_identify = st.checkbox('Identify Outliers')
    checkbox_remove = st.checkbox('Remove Outliers')
    if checkbox_identify:
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
                    st.write('For the variable {}, No of Outliers is {}'.format(each_variable, len(outliers)))


        outliers(df[continuous_variables], drop=False)

    if checkbox_remove:
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
                if drop:
                    df.drop(outliers, inplace=True, errors='ignore')
                    st.write('{} Outliers from {} variable removed'.format(len(outliers), each_variable))


        outliers(df[continuous_variables], drop=True)

    checkbox_updated = st.checkbox('Updated Data')
    if checkbox_updated:
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
                if drop:
                    df.drop(outliers, inplace=True, errors='ignore')


        outliers(df[continuous_variables], drop=True)
        fig, axes = plt.subplots(figsize=(20, 20))
        df.plot(kind='box', subplots=True, layout=(5, 5), sharex=False, sharey=False, ax=axes, color='deeppink')
        st.pyplot(fig)
        st.write('Data shape without outliers:', df.shape)

if eda == 'Distribution of Values':
    fig, ax = plt.subplots(figsize=(5, 4))
    name = ["Team 1 Win", "Team 2 Win"]
    ax = df["outcome"].value_counts().plot(kind='bar')
    ax.set_title("Match Outcomes", fontsize=13, weight='bold')
    ax.set_xticklabels(name, rotation=0)

    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
    total = sum(totals)
    for i in ax.patches:
        ax.text(i.get_x() + .09, i.get_height() - 50,
                str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=14,
                color='white', weight='bold')

    st.pyplot(fig)

if eda == 'Variable Correlation':
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns
    df_numeric = df[numeric_columns]

    df = df.select_dtypes(include='number').columns
    st.header('Correlation Between Variables')
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df_numeric.corr(), ax=ax, annot=True, linewidths=0.01, cmap="Blues")
    st.write(fig)

if eda == 'Dataset Profile Report':
    st.title('2019 Womens World Cup')
    pr = df.profile_report()
    st_profile_report(pr)


# References:
# Add text to plot - https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
# CalculatingTeam1&2CorrelationCoefficient-https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
# NumPy arrange - Generating array of indicies - https://numpy.org/doc/stable/reference/generated/numpy.arange.html
# Matplotlib axex set x_ticks - specify locations https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html
# Matplotlib ax.tick_params - modify pins in plot - https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
# Matplotlib axes legend - shows lables that match w diff elements https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
# Matplotlib autopct - https://matplotlib.org/stable/contents.html
# Matplotlib common string operations cheat sheet - https://matplotlib.org/stable/contents.html
# Plotly box plot - https://plotly.com/python/box-plots/
# Seaborn KDE histogram layer - https://seaborn.pydata.org/tutorial/distributions.html
# st.cache:
# Streamlit's official documentation on st.cache: https://docs.streamlit.io/library/api-reference/memory-caching/st.cache
# A blog post by Streamlit on how to use caching: https://blog.streamlit.io/caching-in-streamlit/
