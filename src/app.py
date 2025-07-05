"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Jan 2025
updated: July 2025
"""
from flask import Flask, redirect
from flask_cors import cross_origin
from preprocess.dataPreprocess import preprocessing
from model.dataTraining import train
from model.modelTracking import tracking

app = Flask(__name__)

# Default endpoint which will redirect to /api
# CORS is used to allow cross-origin requests
# @app.route('/', methods=['GET'])
# @cross_origin()
# def default():
#     return redirect("/api")

# To check /api status
@app.route("/api", methods=['GET'])
def status():
    return {
            'status': 'Success',
            'method': 'GET',
            'message': 'APIs are up and running',
            }
    
# To fetch /model metadata
@app.route("/api/preprocess", methods=['GET'])
def model_metadata_api():
    metadata = preprocessing()
    if metadata is None:
        return {
            'status': 'Error',
            'method': 'GET',
            'message': 'No preprocessed data found.'
        }, 404
    return metadata

# To /train model performance
@app.route("/api/model/train", methods=['GET'])
def model_train_api():
    metadata = train()
    if metadata is None:
        return {
            'status': 'Error',
            'method': 'GET',
            'message': 'No model training data found.'
        }, 404
    return metadata

# To /track model performance
@app.route("/api/model/track", methods=['GET'])
def model_track_api():
    metadata = tracking()
    if metadata is None:
        return {
            'status': 'Error',
            'method': 'GET',
            'message': 'No model tracking data found.'
        }, 404
    return metadata

# # To /test model performance
@app.route("/api/model/test", methods=['GET'])
def model_test_api():
    test_model = "model_testing()"
    return test_model

# To allow public and private access to the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)