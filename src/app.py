"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Jan 2025
updated: May 2025
"""
from flask import Flask, redirect
from flask_cors import cross_origin
from preprocess.dataPreprocess import preprocessing
# from model.model_training import train
# from model.model_tracking import model_tracking, model_testing

app = Flask(__name__)

# Default endpoint which will redirect to /api
# CORS is used to allow cross-origin requests
@app.route('/', methods=['GET'])
@cross_origin()
def default():
    return redirect("/api")

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
    return metadata

# # To /track model performance
# @app.route("/api/model/track", methods=['GET'])
# def model_track_api():
#     metadata = "model_tracking()"
#     return metadata

# # To /test model performance
# @app.route("/api/model/test", methods=['GET'])
# def model_test_api():
#     test_model = "model_testing()"
#     return test_model

# To allow public and private access to the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)