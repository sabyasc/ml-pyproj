"""
author: sabyasc
github: https://github.com/sabyasc
created: Dec 2024
"""
from flask import Flask
from models.train import data_ingestion

app = Flask(__name__)

@app.route("/api", methods = ['GET'])
def status():
    return {'status': 'Success'}

@app.route("/api/model", methods = ['GET'])
def pre_process_api():
    process_data = data_ingestion()
    return process_data

if __name__ == '__main__':
    app.run(debug=True, port='5000')