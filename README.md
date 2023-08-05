# <a name="top"></a>Dog Bite Forecasting

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
    * Use datetime weekly resampling to explore diffent instances.
        * Are there any noticeable trends in the number of dog bites over time?
        * Do dog bites exhibit seasonal patterns? Are certain months or seasons associated with a higher number of incidents?
        * Are there specific days of the week when dog bites are more frequent than others? Do weekends or weekdays show different patterns?
        * What impact does dog bite have in differnt locations?
  
  * Model with the following timeseries models.
  
  - Moving average
  - Holt linear model
  - Holt seasonal model
  - Previous cycle
  
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
| bite           | The number of dog bite incidents reported for a day. | Integer  |

# Steps to Reproduce
  * Clone this repo
  * Run the aquire.ipynb and prepare.ipynb (Must to download necessary files)
  * Run the final_report notebook

# Conclusions
- I failed to beat my baseline model (Moving Average), by 1 weekly bites. This baseline model performed with a 16 RMSE and my closes non best line model (Holts Seasonal Trend) had a 17 RMSE.

# Next Steps
- College more data then achieve a working model on the weekly or monthly resamples.
- How long after getting bitten did they wait before receiving medical attention?
- What effect does the dog's age have on bites?

# Recommendations
- Do not use this model to forecast changes in weekly dog bites, however, I recommend the local authorities to remain diligent and on high alert during the weekend and especially more flexible in the summer because that's when the data shows high bite rates. 

- I also recommend for necessary resources to distributed mainly in Queens, Manhattan, and Brooklyn to help with incoming emergencies.