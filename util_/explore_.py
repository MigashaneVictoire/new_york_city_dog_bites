# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# seasonal decomposition
import statsmodels.api as sm

# system manipulation
import itertools
import os
import sys
sys.path.append("./util_")
import prepare_
# import explore_

# other
import warnings
warnings.filterwarnings("ignore")

# set the random seed
np.random.seed(95)


# set a default them for all my visuals
sns.set_theme(style="whitegrid")

# _______________________ Set up___________________________________
# load data from existing csv
dogs_train = pd.read_csv("./00_project_data/1-1_training_data.csv", index_col=0)

# make sure the index in in time format
dogs_train.index = dogs_train.index.astype("datetime64")

train = dogs_train.resample('D').bite.sum()
train = pd.DataFrame(train)

# create a year, month, day column from the index
train["year"] = train.index.year
train["month"] = train.index.month_name()
train["week day"] = train.index.day_name()
train["day of month"] = train.index.day
train["month number"] = train.index.month
train["week day number"] = train.index.weekday

# _____________________Explore Functions____________________________________

def get_trend1():
    """
    Return plot for answering the following quesion:
        - Are there any noticeable trends in the number of dog bites over time?
    """
    
    plt.figure(figsize=(8,4))
    train.bite.resample('D').mean().plot(alpha=.5, label='Daily')
    train.bite.resample('W').mean().plot(alpha=.8, label='Weekly')
    train.bite.resample('M').mean().plot(label='Montly')
    train.bite.resample('Y').mean().plot(label='Yearly')
    plt.title("General dog bite trend over time")
    plt.xlabel('date of bite')
    plt.ylabel('Frequency')
    plt.legend()
    return plt.gcf()

def get_high_month():
    """
    Return plot for answering the following quesion:
        - Are certain months or seasons associated with a higher number of incidents?
    """
    # name of month to choose from
    x_labes = x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors = ['gray', 'gray', 'gray', 'orange', 'red', 'red', 'red', 'red', 'orange', 'orange', 'orange', 'gray']

   # Yearly mean bits
    plt.figure(figsize=(7,3))
    train.groupby("month number").bite.mean().plot(kind="bar", color=colors)
    plt.title("Monthly mean bite")
    plt.xticks(range(12), x_labels)
    plt.xlabel("month")
    plt.ylabel("bite")
    plt.xticks(rotation=0)
    return plt.gcf()

def get_week_diff():
    """
    Return plot for answering the following quesion:
        - Are there specific days of the week when dog bites are more frequent than others?
    """
    # name of month to choose from
    x_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    colors = ['gray', 'gray', 'gray', 'gray', 'gray', 'red', 'red']
    # Yearly mean bits
    plt.figure(figsize=(7,3))
    train.groupby("week day number").bite.mean().plot(kind="bar", color=colors)
    plt.title("Weekday mean bites")
    plt.xticks(range(7), x_labels)
    plt.xlabel("week day")
    plt.ylabel("bite")
    plt.xticks(rotation=0)
    return plt.gcf()

def get_borough():
    """
    Return plot for answering the following quesion:
        - What impact does dog bite have in differnt locations?
    """
    colors = ['red', 'red', 'red', 'gray', 'gray', 'gray']
    # borough of bite
    plt.figure(figsize=(7, 3))
    dogs_train.borough.value_counts().plot.bar(color=colors)
    plt.title("bite count by borough")
    plt.xlabel("week day")
    plt.ylabel("total bites")
    plt.xticks(rotation=0)
    return plt.gcf()