# <a name="top"></a>Dog Bite Forcasting

by: Victoire Migashane

<p>
  <a href="https://github.com/MigashaneVictoire" target="_blank">
    <img alt="Victoire" src="https://img.shields.io/github/followers/MigashaneVictoire?label=Follow_Victoire&style=social" />
  </a>
</p>
---

# Project Description
This project aims to build a model that predicts dog bite for the city of New York.

# Project Goal
To predict the number of dog bites in New York City for the next week with reasonable accuracy. Because it takes about 2 to 10 days to die from the infection with no medical attention.

**Audience and importance**

This goal is relevant because it can help authorities and relevant stakeholders to prepare for potential increases in dog bites and allocate resources accordingly to mitigate risks.

# Initial Thoughts
Do to the stracture of the data, I will need to feature engineer a column to represent my bites.

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
    * Use datetime resampling to explore diffent instances.
        * Are there any noticeable trends in the number of dog bites over time?
        * Do dog bites exhibit seasonal patterns? Are certain months or seasons associated with a higher number of incidents?
        * Are there specific days of the week when dog bites are more frequent than others? Do weekends or weekdays show different patterns?
        * **What impact does dog bite have in differnt locations?
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
| borough     | Borough in New York City where the incident occurred. | String    |
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
  * I was not able to beat my base model (moving average) which resulted in a () rmse and predicting with in () range of the actual bite couts weekly.

# Next Steps
- Was the dog a neutered or not, to check the impact of time before medical attention.
- What affect does the dogs age have on bites

# Recommendations
- Use my model if you want to find bi-weekly changes in dog bite.
- Local authorities to be on high alert in the (boroughs) and alocate more resorces in these rejons of the city.
- To avoid giving random rabies choots after dog bite, it's important for the local authorities to on high alert, but it's also the resonsibility for the person beat to report the bite sooner to avoid the spread of the infection.
  