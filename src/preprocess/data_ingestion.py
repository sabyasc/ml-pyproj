"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Dec 2024
"""
import os, requests
import pandas as pd

# Data Ingestion is to fetech data from sources, We will follow below steps:
# Step 1: Read data from source (csv, json, db, APIs, etc) using os and requests,
# Step 2: Display first 10 example rows of data to understand the data
def ingestion():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_url = "https://jsonplaceholder.typicode.com/posts"
    api_response = requests.get(api_url)
    
    if api_response.ok:
        api_data = api_response.json()
        print("================ API Data Ingestion Completed ================")
    else:
        api_data = {}
        print("Failed to fetch data from API.")
        
    data_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data', 'raw', 'train.csv')
    data = pd.read_csv(data_path)

    df = pd.DataFrame(data)
    count = df.head(50).to_dict()
    print("================ CSV Data Ingestion Completed ================")
    return count
