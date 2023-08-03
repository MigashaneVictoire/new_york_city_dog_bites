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
import itertools

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
def evaluate(target, model_type= 0, model_df= None, y_pred_df=None):
    """
    Takes in the target col name and return the root mean square error for
    the validate/test against the predicted values.
    """
    if model_type == 1:
        rmse = round(sqrt( # get root mean square error
        mean_squared_error(validate[target], model_df[target])),3) # get mean squared error
        return rmse
    else:
        rmse = round(sqrt( # get root mean square error
            mean_squared_error(validate[target], y_pred_df[target])),3) # get mean squared error
        return rmse
    
# plot predictions vs actual values
def plot_evaluate(target, model_type= 0, model_df=None, y_pred_df=None):
    """
    Function will return a plot of the predicted values against the actual values in the data.
    """
    if model_type == 1:
        # plt.ioff()
        # plt the value value fof bite over time and the predicted
        plt.figure(figsize=(8,3))
        plt.plot(train.bite, label="Train", linewidth=1)
        plt.plot(validate.bite,label="Validate", linewidth=1)
        plt.plot(model_df[target],label="Prediction", linewidth=1 )
        plt.title("Dog bite over time prediction")
        plt.legend()

        rmse = evaluate(target, model_type= 1, model_df= model_df)
        print(target, f'-- rmse: {round(rmse,2)}')
        return rmse
    else:
        # plt.ioff()
        # plt the value value fof bite over time and the predicted
        plt.figure(figsize=(8,3))
        plt.plot(train.bite, label="Train", linewidth=1)
        plt.plot(validate.bite,label="Validate", linewidth=1)
        plt.plot(y_pred_df[target],label="Prediction", linewidth=1 )
        plt.title("Dog bite over time prediction")
        plt.legend()

        rmse = evaluate(target, y_pred_df=y_pred_df)
        print(target, f'-- rmse: {round(rmse,2)}')
        return rmse

# function to append to the evaluation dataframe
def append_to_eval(model, target, y_pred_df=None):
    """
    Function append the current running model results the historical models.
    """
    rmse = evaluate(target,y_pred_df=y_pred_df)
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

# ____________________________________________________________

def get_moving_average():
    """
    Goal: Run the moving avearge model and return the viual with rmse
    """
    # looking at 2% of the validation data
    lags = math.ceil(validate.shape[0] * .3)

    predictions_dfs = []
    for i in range(1, lags + 1):
        # get the average over a lot of different weeks
        rolling_avg = round(train.bite.rolling(i).mean()[-1],2)
        
        # predict baseline over time
        y_pred_df = pd.DataFrame({"bite": rolling_avg}, index=validate.index)
        
        # add to moving average predictions dataframes
        predictions_dfs.append(y_pred_df)
        
        # update the evaluate dataframe
        model_type = str(i) + '-wk moving avg'
        eval_df= append_to_eval(model=model_type, target="bite", y_pred_df=y_pred_df)

    # base model rmse
    mv_avg_rmse_df = eval_df.reset_index(drop=True)
    mv_avg_rmse_df[mv_avg_rmse_df.rmse == mv_avg_rmse_df.rmse.min()]

    # find best baseline model
    mov_avg_pred = pd.DataFrame(predictions_dfs[22], index=validate.index)

    # Plot prediction
    plot_evaluate(target= "bite", model_type= 1, model_df= predictions_dfs[22], y_pred_df=y_pred_df)
    
    return plot_evaluate


    # Plot prediction
    plot_evaluate(target= "bite", model_type= 1, model_df= predictions_dfs[22])
    plt.show()

def get_holt_linear_trend():
    """
    Goal: Run the holt linear trend model and return the viual with rmse
    """
    # create prediction dataframe
    y_pred_df = pd.DataFrame(columns=["bite"])

    # create object fit and predict
    model = Holt(train["bite"], exponential=False, damped=True)
    model = model.fit(optimized=True)

    y_pred_values = model.predict(start=validate.index[0],
                                end= validate.index[-1])
    y_pred_df["bite"] = round(y_pred_values, 3)

    eval_df = append_to_eval(model = 'holts_optimized', target= "bite", y_pred_df=y_pred_df)

    # Plot prediction
    plot_evaluate(target= "bite", y_pred_df=y_pred_df) # this returns the rsme for the plot for evaluation
    plt.show()

def get_holt_seasonal_trend():
    """
    Goal: Run the holt seasonal trend model and return the viual with rmse
    """
    # create evaluation dataframe
    eval_df = pd.DataFrame(columns=["model", "rmse"])

    # create prediction dataframe
    y_pred_df = pd.DataFrame(columns=["bite"])

    # get combinations to pass to model object
    combos = list(itertools.product(['add','mul'],[True, False]))

    # # Models for quantity
# # 52 because there is 52 weeks in a year and our data has been sampled by week
    hsts = {}

    for i, combo in enumerate(combos):
        hsts['hst_fit_' + str(i)] = ExponentialSmoothing(train.bite, seasonal_periods=52, trend='add', seasonal=combo[0], damped=combo[1]).fit()
    
    hsts_score_eval_df = {}
    # compare our rmse
    for model, obj in hsts.items():
        score = sqrt(hsts[model].sse / len(train)) # compute the evaluation score (chose the smallest as best to run predictions)
        hsts_score_eval_df[model] = [round(score, 3)]

    # assign the prediction results to a dataframe
    y_pred_df = pd.DataFrame(hsts['hst_fit_0'].forecast(validate.shape[0]), columns=["bite"])

    # Plot prediction
    best_hsts_rmse = plot_evaluate(target= "bite", y_pred_df=y_pred_df) # this returns the rsme for the plot for evaluation

    # # add current model rmse to evaluation dataframe
    # eval_df= pd.concat([eval_df.reset_index(drop=True), current_model], axis=0)    


    # Create a new row as a dictionary
    hsts_rmse = {'model':'hst_fit_0',
                'rmse':best_hsts_rmse}

    # Append the new row to the existing data frame
    eval_df = eval_df.append(hsts_rmse, ignore_index=True)
    plt.show()

def get_previous_cycle():
    """
    Goal: Run the previous cycle model and return the viual with rmse
    """
    # create prediction dataframe
    y_pred_df = pd.DataFrame(columns=["bite"])

    # Calculates the difference of a DataFrame element compared with another element
    # period (lag) (must be th lenght of the validation season)
    mean_diff = train.diff(periods=52 * 2).mean()

    # find the diff. add to each value in 2015.
    y_pred_df = train['2016':'2017'] + mean_diff

    # set yhat_df to index of validate
    y_pred_df.index = validate.index

    # add results to the evaluation dataframe
    plot_evaluate(target= "bite", y_pred_df=y_pred_df)
    eval_df = append_to_eval(model = 'previous year', target = "bite", y_pred_df=y_pred_df)

def get_test_results():
    """
    Goal: Run the moving avearge best model and return the viual with rmse
    """
    # Final Test
    # get best model average
    rolling_avg = round(validate.bite.rolling(23).mean()[-1],3)

    # make predictions
    y_final_pred_df = pd.DataFrame({"bite": rolling_avg}, index=test.index)

    # update the evaluate dataframe
    model_type = '23-wk moving avg'
    rmse = round(sqrt(mean_squared_error(test["bite"], y_final_pred_df["bite"])),3) # root get mean squared error
    
    # Plot test results
    plt.figure(figsize=(8,3))
    # plt.plot(train.bite, label="Train", linewidth=1)
    plt.plot(validate.bite, label="Train", linewidth=1)
    plt.plot(test.bite,label="Validate", linewidth=1)
    plt.plot(y_final_pred_df['bite'],label="Prediction", linewidth=1 )
    plt.title("Dog bite test prediction")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    return rmse