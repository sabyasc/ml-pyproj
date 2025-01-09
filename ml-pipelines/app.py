"""
author: sabyasc
github: https://github.com/sabyasc
created: Dec 2024
"""
from flask import Flask
from models.train import model_testing, model_deployment, model_tracking

app = Flask(__name__)

# To check /api status
@app.route("/api", methods = ['GET'])
def status():
    return {'status': 'Success'}

# To fetch /model metadata
@app.route("/api/model", methods = ['GET'])
def model_metadata_api():
    metadata = model_deployment()
    return metadata

# To /track model performance
@app.route("/api/model/track", methods = ['GET'])
def model_track_api():
    metadata = model_tracking()
    return metadata

# To /test model performance
@app.route("/api/model/test", methods = ['GET'])
def model_test_api():
    test_model = model_testing()
    return test_model

if __name__ == '__main__':
    app.run(port=5000)