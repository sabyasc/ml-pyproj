"""
author: sabyasc
github: https://github.com/sabyasc
created: Dec 2024
"""
from flask import Flask
from models.train import model_validation, model_testing

app = Flask(__name__)

# To check /api status
@app.route("/api", methods = ['GET'])
def status():
    return {'status': 'Success'}

# To fetch /api/model details
@app.route("/api/model", methods = ['GET'])
def data_processing_api():
    processed_data = model_validation()
    return processed_data

# To fetch /api/model/test details
@app.route("/api/model/test", methods = ['GET'])
def data_testing_api():
    test_data = model_testing()
    return test_data

if __name__ == '__main__':
    app.run(debug=True, port='5000')