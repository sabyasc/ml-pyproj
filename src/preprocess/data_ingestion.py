"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Dec 2024
"""
import os, requests, pandas as pd

# Data Ingestion is to fetech data from sources, We will follow below steps:
# Step 1: Read data from source (csv, json, db, APIs, etc) using os and requests,
# Step 2: Choose either API or CSV to read data and display first 10 example rows of data,
# Step 3: Return the data to the calling function,
# Step 4: (Optional) Print the message of data ingestion completion 
def ingestion():
    choice = input("Choose data source: 1 for API, 2 for CSV: ")
    
    if choice == "1":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_url = "https://jsonplaceholder.typicode.com/posts"
        api_response = requests.get(api_url)
    
        if api_response.ok:
            api_data = api_response.json()
            print("================ API Data Ingestion Completed ================")
            result = api_data[:10]
        else:
            print("Failed to fetch data from API.")
            result = {}
            
    elif choice == "2":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data', 'raw', 'train.csv')
        data = pd.read_csv(data_path)
    
        df = pd.DataFrame(data)
        result = df.head(50).to_dict()
        print("================ CSV Data Ingestion Completed ================")
        
    else:
        print("Invalid choice. Please choose either 1 or 2.")
        result = None
        
    return result

# Uncomment to test the function and see the output
# print(ingestion())
