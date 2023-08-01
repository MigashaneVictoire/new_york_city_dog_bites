# For funtion annotations
from binascii import a2b_qp
from typing import Union
from typing import Tuple

# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# data separation/transformation
from sklearn.model_selection import TimeSeriesSplit

# system manipulation
import os
import sys
# sys.path.append("./util_")
import acquire_

# other
from math import ceil
import warnings
warnings.filterwarnings("ignore")

# set a default them for all my visuals
sns.set_theme(style="whitegrid")

#############################################################################################
def wrangle_dog_bites():
    """
    Goal: Complet preparation for the dogs bite data
    """
    # get data from acquire file
    dogs = acquire_.get_dogs_data()
    
    # make the columsn lower case
    # replace spaces
    dogs.columns = dogs.columns.str.lower().str.strip().str.replace("†","").str.replace(" ","_")
    
    # convert string to datetime
    dogs.dateofbite = dogs.dateofbite.astype("datetime64")
    
    # remove duplicated rows
    dogs = dogs.drop_duplicates()
    
    # extract date column and remove everything else
    dogs = dogs[dogs.columns[dogs.columns.isin(["dateofbite"])]]
    
    # create my target bite column
    dogs["bite"] = int(1)
    
    # set the date to be the indexinf column
    dogs = dogs.set_index("dateofbite").sort_index()
    
    # Engineer the target into daily bits counts
    dogs = dogs.resample('D').sum()
    
    # create a year, month, day column from the index
    dogs["year"] = dogs.index.year
    dogs["month"] = dogs.index.month_name()
    dogs["day"] = dogs.index.day_name()

    dogs["month_number"] = dogs.index.month
    dogs["weekday_number"] = dogs.index.weekday
    
    # split data
    train = dogs[:"2018"]
    validate = dogs["2019":"2020"]
    test = dogs["2021":]
    
    # save the split data
    prepare_.save_split_data_(cleaned_data=dogs, train=train, validate=validate, test=test)

    return train, validate, test

#---------------------------------------------------------------
# Save visuals
def save_visuals_(fig: plt.figure ,viz_name:str= "unamed_viz", folder_name:int= 0, ) -> str:
    """
    Goal: Save a single visual into the project visual folder
    parameters:
        fig: seaborn visual figure to be saved
        viz_name: name of the visual to save
        folder_name: interger (0-7)represanting the section you are on in the pipeline
            0: all other (defealt)
            1: univariate stats
            2: bivariate stats
            3: multivariate stats
            4: stats test
            5: modeling
            6: final report
            7: presantation
    return:
        message to user on save status
    """
    project_visuals = "./00_project_visuals"
    folder_selection = {
        0: "00_non_specific_viz",
        1: "01_univariate_stats_viz",
        2: "02_bivariate_stats_viz",
        3: "03_multivariate_stats_viz",
        4: "04_stats_test_viz",
        5: "05_modeling_viz",
        6: "06_final_report_viz",
        7: "07_presantation"
    }

    # return error if user input for folder selection is not found
    if folder_name not in list(folder_selection.keys()):
        return f"{folder_name} is not a valid option for a folder name."
    # when folder location is found in selections
    else:
        # Specify the path to the directory where you want to save the figure
        folder_name = folder_selection[folder_name]
        directory_path = f'{project_visuals}/{folder_name}/'

        # Create the full file path by combining the directory path and the desired file name
        file_path = os.path.join(directory_path, f'{viz_name}.png')

        if os.path.exists(project_visuals): # check if the main viz folder exists
            if not os.path.exists(directory_path): # check if the folder name already exists
                os.makedirs(directory_path)
                # Save the figure to the specified file path
                fig.canvas.print_figure(file_path)

            else:
                # Save the figure to the specified file path
                fig.canvas.print_figure(file_path)
        else:
            # create both the project vis folder and the specific section folder
            os.makedirs(project_visuals)
            os.makedirs(directory_path)

            # Save the figure to the specified file path
            fig.canvas.print_figure(file_path)
    
    return f"Visual successfully saved in folder: {folder_name}"

# -----------------------------------------------------------------
# Save the splited data into separate csv files
def save_split_data_(cleaned_data: pd.DataFrame, train:pd.DataFrame, validate:pd.DataFrame, test:pd.DataFrame,
                      folder_path: str = "./00_project_data") -> str:
    """
    parameters:
        cleaned_data: full project dataframe that contains the (encoded columns or scalling)
        train: training data set that has been split from the original
        validate: validation data set that has been split from the original
        test: testing data set that has been split from the original
        folder_path: folder path where to save the data sets

        Only apply to spliting the original_df in inside this function
            --> test_size:float = 0.2,stratify_col:str=None, random_state: int=95
    return:
        string to show succes of saving the data
    """
    # get original data to be saved
    original_df = acquire_.get_org_data()

    # create new folder if folder don't aready exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        # save the dataframe original
        original_df.to_csv(f"./{folder_path}/1-0_original_cleaned_data.csv", mode="w")

        # save the dataframe with dummies in a csv for easy access
        cleaned_data.to_csv(f"./{folder_path}/1-0_original_cleaned_data.csv", mode="w")

        # save training data
        train.to_csv(f"./{folder_path}/1-1_training_data.csv", mode="w")

        # save validate
        validate.to_csv(f"./{folder_path}/1-2_validation_data.csv", mode="w")

        # Save test
        test.to_csv(f"./{folder_path}/1-3_testing_data.csv", mode="w")

    else:
         # save the dataframe original
        original_df.to_csv(f"./{folder_path}/1-0_original_cleaned_data.csv", mode="w")

        # save the dataframe with dummies in a csv for easy access
        cleaned_data.to_csv(f"./{folder_path}/1-0_original_cleaned_data.csv", mode="w")

        # save training data
        train.to_csv(f"./{folder_path}/1-1_training_data.csv", mode="w")

        # save validate
        validate.to_csv(f"./{folder_path}/1-2_validation_data.csv", mode="w")

        # Save test
        test.to_csv(f"./{folder_path}/1-3_testing_data.csv", mode="w")

    return "Four data sets saved as .csv"


# -----------------------------------------------------------------
# Split the data into train, validate and train
def split_data_(df: pd.DataFrame, test_size: float =.2, validate_size: float =.2, stratify_col: str =None, random_state: int=95) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    parameters:
        df: pandas dataframe you wish to split
        test_size: size of your test dataset
        validate_size: size of your validation dataset
        stratify_col: the column to do the stratification on
        random_state: random seed for the data

    return:
        train, validate, test DataFrames
    '''
    # no stratification
    if stratify_col == None:
        # split test data
        train_validate, test = train_test_split(df, 
                                                test_size=test_size, 
                                                random_state=random_state)
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                            random_state=random_state)
    # stratify split
    else:
        # split test data
        train_validate, test = train_test_split(df,
                                                test_size=test_size,
                                                random_state=random_state, 
                                                stratify=df[stratify_col])
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                           random_state=random_state, 
                                           stratify=train_validate[stratify_col])
    return train, validate, test