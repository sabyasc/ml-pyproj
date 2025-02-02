"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Dec 2024
"""
import os
import pandas as pd

# Data Ingestion is to fetech data from sources, We will follow below steps:
# Step 1: Read data from source (csv, json, db, APIs, etc) using os and pandas,
# Step 2: Display first 10 example rows of data to understand the data
def ingestion():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data', 'raw', 'train.csv')
    data = pd.read_csv(data_path)
    df = pd.DataFrame(data)
    count = df.head(50).to_dict()
    print("================ Data Ingestion completed ================")
    return count
