"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Dec 2024
"""
# src libs are fetched from __init__.py
from src import pd

# Data Ingestion is to fetech data from sources, We will follow below steps:
# Step 1: Read data from source (csv, json, db, APIs, etc),
# Step 2: Display first 10 example rows of data to understand the data,
# Step 3: Return the data as Dataframe to next step for processing
def ingestion():
    data = pd.read_csv(r'C:\GitHub\ml-pyproj\data\raw\train.csv')
    df = pd.DataFrame(data)
    count = df.head(50)
    print("================ Data Ingestion completed ================")
    return count
