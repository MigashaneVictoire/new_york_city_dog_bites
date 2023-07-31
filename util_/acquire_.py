# Python libraries
import pandas as pd
import requests

# be sure to name this function the same name
def get_org_data() -> pd.DataFrame:
    """
    Goal: Retrieve the failed bank data from data.gov
    """
    # get data with csv link
    data_url = "https://data.cityofnewyork.us/api/views/rsgh-akpg/rows.csv"
    
        # Try different encodings one by one until one works without an error
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'utf-16']

    for encoding in encodings_to_try:
        try:
            # Read the CSV file from the URL with the specified encoding
            df = pd.read_csv(data_url, index_col= 0, encoding=encoding)
            # If no error occurs, break the loop and continue with further processing
            break
        except UnicodeDecodeError:
            # If an error occurs with the current encoding, try the next one
            continue

    return df
