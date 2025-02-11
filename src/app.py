"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Jan 2025
"""
from flask import Flask, redirect
from preprocess.data_ingestion import ingestion
# from model.model_training import train
# from model.model_tracking import model_tracking, model_testing

app = Flask(__name__)

# Default endpoint which will redirect to /api
@app.route('/', methods=['GET'])
def default():
    return redirect("/api")

# To check /api status
@app.route("/api", methods=['GET'])
def status():
    return {
            'status': 'Success',
            'message': 'Please use /api/model, /api/model/track, /api/model/test endpoints for your requests'
            }

# # To fetch /model metadata
@app.route("/api/model", methods=['GET'])
def model_metadata_api():
    metadata = ingestion()
    return metadata

# # To /track model performance
# @app.route("/api/model/track", methods=['GET'])
# def model_track_api():
#     metadata = model_tracking()
#     return metadata

# # To /test model performance
# @app.route("/api/model/test", methods=['GET'])
# def model_test_api():
#     test_model = model_testing()
#     return test_model

if __name__ == "__main__":
    app.run(port=5000)
