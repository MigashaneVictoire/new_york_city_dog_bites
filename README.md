# <a name="top"></a>Dog Bite Forcasting

by: Victoire Migashane

<p>
  <a href="https://github.com/MigashaneVictoire" target="_blank">
    <img alt="Victoire" src="https://img.shields.io/github/followers/MigashaneVictoire?label=Follow_Victoire&style=social" />
  </a>
</p>
---

# Project Description
This project aims to build a model using tha predicts dog bites for the city of New York.

# Project Goal
To predict the number of dog bites in New York City for the next 3-months with reasonable accuracy.

**Audience and importance**

This goal is relevant because it can help authorities and relevant stakeholders to prepare for potential increases in dog bites and allocate resources accordingly to mitigate risks.

# Initial Thoughts
Due to limited domain knowledge of the chemical properties of wine, feature selection may be the best option in selecting most important features for the model. 

# The Plan
  * Acquire data from Data.World
    - Data from [data.gov](https://catalog.data.gov/dataset/dohmh-dog-bite-data)
    - I have 22663 rows and 9 columns.
    - 8 columns are objects and 1 1 boolean.
    - 0 null count.
    
  * Prepare data
    - Rename columns.
    - Convert data types (dateofbite).
    - Drop columns (all except dateofbite).
    - Drop duplicates (541, 2%).
    - Set and sort date index.
    - Check for gaps in time (0 days missing).
    - Create a dog bite target column.
    - Engineer the target into daily bits counts.
    - Use human-based splitting.
      
  * Explore the data
    * Use Recursive Feature Elimination to determine best features for wine quality
    * Answer the questions:
        * Does the average quality score differ between red or white wines?
        * Is there a relationship between volatile acidity and density?
        * What does clustering show us about this correlation?
        * Is there a relationship between density and alcohol level?
        * Is there a relationship between volatile acidity and free sulfur dioxide?
        
  * Develop a model to predict wine quality score
    * Use accuracy as my evaluation metric.
    * Baseline will be the mode of quality score.
    * Target variable was binned to improve model performance.
        * Low Score Wine: Quality Score of 5 and Lower
        * High Score Wine: Quality Score of 6 and Higher
   
  * Make conclusions.

# Data Dictionary
#### DOHMH dog bite data dictionary

The following is a data dictionary for the columns in the DOHMH dog bite dataset:

| Column Name | Description                                     | Data Type |
|-------------|-------------------------------------------------|-----------|
| dateofbite  | Date when the dog bite incident was reported.   | Date      |
| species     | Species of the dog involved in the incident.   | String    |
| breed       | Breed of the dog involved in the incident.     | String    |
| age         | Age of the dog involved in the incident.       | Integer   |
| gender      | Gender of the dog involved in the incident.    | String    |
| spayneuter  | Indicates if the dog is spayed/neutered.       | Boolean   |
| borough     | Borough in New York City where the incident occurred. | String    |
| zipcode     | Zip code of the location where the incident occurred. | Integer   |

#### After Preparation

| Column Name    | Description                                          | Data Type |
|----------------|------------------------------------------------------|-----------|
| bite           | The number of dog bite incidents reported for a day. | Integer   |
| year           | The year of the dog bite incident.                  | Integer   |
| month          | The name of the month of the dog bite incident.     | String    |
| day            | The day of the month of the dog bite incident.      | Integer   |
| month_number   | The numeric representation of the month (1 to 12).  | Integer   |
| weekday_number | The numeric representation of the day of the week (0: Monday, 6: Sunday). | Integer   |

# Steps to Reproduce
  * Clone this repo
  * Acquire the data
  * Put the data in the same file as cloned repo
  * Run the final_report notebook

# Conclusions
  * Decision Tree is the best performing model with approximately 72% accuracy on unseen data.
  * However, this accuracy only applies to the broad categories of low quality wine (quality score of < = 5) and high quality wine (quality score of > = 6)

# Next Steps
  * Develop a model that can predict a specific quality score. 

# Recommendations
  * If the current target groups of low vs high quality wine is sufficient, implement this model.
  * Search for data that is from California wines because the location where the grapes and wine is produced may impact the physiochemical properties.
  