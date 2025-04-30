"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Dec 2024
updated: April 2025
"""

# Data Ingestion is to fetch data from sources, We will follow below steps:
# Step 1: Read data from source (csv, json, db, APIs, etc) using os and requests,
# Step 2: Choose either API or CSV to read data and display first 10 example rows of data,
# Step 3: Return the data to the calling function
def ingestion():
    choice = input("Choose data source: 1 for API, 2 for CSV: ")
    print(f"Choice: {choice}")
