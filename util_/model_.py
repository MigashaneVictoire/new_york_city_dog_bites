# for presentation purposes
import warnings
warnings.filterwarnings("ignore")

# transform
import numpy as np
import pandas as pd

# visualize 
import matplotlib.pyplot as plt
import seaborn as sns

# working with dates
from datetime import datetime

# modeling
import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing

# evaluate
from sklearn.metrics import mean_squared_error
from math import sqrt 

np.random.seed(95)
import math

# load data from existing csv
train = pd.read_csv("./00_project_data/1-1_training_data.csv", index_col=0)
validate = pd.read_csv("./00_project_data/1-2_validation_data.csv", index_col=0)
test = pd.read_csv("./00_project_data/1-3_testing_data.csv", index_col=0)

# covert the index to datetime
train.index = train.index.astype("datetime64")
validate.index = validate.index.astype("datetime64")
test.index = test.index.astype("datetime64")

# resample to only weekly data
train = train.resample('w').bite.sum()
validate = validate.resample('w').bite.sum()
test = test.resample('w').bite.sum()
train = pd.DataFrame(train)
validate = pd.DataFrame(validate)
test = pd.DataFrame(test)

# create evaluation dataframe
eval_df = pd.DataFrame(columns=["model", "rmse"])

# evaluation funtion
def evaluate(target):
    """
    Takes in the target col name and return the root mean square error for
    the validate/test against the predicted values.
    """
    rmse = round(sqrt( # get root mean square error
        mean_squared_error(validate[target], y_pred_df[target])),3) # get mean squared error
    return rmse
    
# plot predictions vs actual values
def plot_evaluate(target):
    """
    Function will return a plot of the predicted values against the actual values in the data.
    """
    # plt.ioff()
    # plt the value value fof bite over time and the predicted
    plt.figure(figsize=(8,3))
    plt.plot(train.bite, label="Train", linewidth=1)
    plt.plot(validate.bite,label="Validate", linewidth=1)
    plt.plot(y_pred_df[target],label="Prediction", linewidth=1 )
    plt.title("Dog bite over time prediction")
    plt.legend()

# function to append to the evaluation dataframe
def append_to_eval(model, target):
    """
    Function append the current running model results the historical models.
    """
    rmse = evaluate(target)
    res_dict = pd.DataFrame({"model":[model],
               "rmse":[rmse]})
    return pd.concat([eval_df, res_dict])

def get_best_model():
    """
    Goal: return the best visual
    """
    md = pd.DataFrame({"Model": ["Moving average 23wk", "Holt linear trend", 
                        "Holt seasonal trend", "Previous cycle"],
              "RMSE":[16, 38, 17, 21]})

    colors = ["blue", "gray", "gray", "gray"]
    plt.figure(figsize=(8,3))
    ax = sns.barplot(data= md, x="Model", y='RMSE', palette=colors)
    plt.title("Model RMSE weekly")

    # Annotate each bar with its corresponding value
    for index, value in enumerate(md['RMSE']):
        ax.text(index, value, str(value), ha='center', va='bottom')
    return plt.gcf()

def get_final_best_model():
    """
    Goal: return the best visual
    """
    md = pd.DataFrame({"Model": ["Moving average 23wk", "Moving average 23wk test"], "RMSE":[16, 15]})

    colors = ["gray", "blue"]
    plt.figure(figsize=(8,3))
    ax = sns.barplot(data= md, x="Model", y='RMSE', palette=colors)
    plt.title("Test 23-weeks Moving Average RMSE")

    # Annotate each bar with its corresponding value
    for index, value in enumerate(md['RMSE']):
        ax.text(index, value, str(value), ha='center', va='bottom')
    return plt.gcf()


