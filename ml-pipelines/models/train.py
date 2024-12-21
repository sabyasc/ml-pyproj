"""
author: sabyasc
github: https://github.com/sabyasc
created: Dec 2024
"""
import pandas as pd

# Data Ingestion is to fetech data from sources
def data_ingestion():
    data = pd.read_csv(r'C:\GitHub\ml-pyproj\ml-pipelines\dataset\train.csv')
    df = pd.DataFrame(data)
    count = df.head(5)
    print("================ Data Ingestion completed ================")
    return count

print(data_ingestion())
