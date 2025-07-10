"""
author: sabyasc
github: https://github.com/sabyasc
created: Dec 2024
updated: July 2025
"""
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from models.train import  ingestion, preprocessing, model_training, model_validation, model_deployment
import timedelta, datetime

import os
import sys

os.getlogin()
# Patch Airflow to avoid using the pwd module on Windows
if sys.platform == 'win32':
    import builtins
    builtins.pwd = None

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ensemble_model_pipeline',
    default_args=default_args,
    description='Classification Model ML Pipeline by @sabyasc',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 12, 23),
    tags=['classification', 'ensemble', 'model', 'pipeline', 'mlops'],
    catchup=False,
) as dag:
    
    ingest_task = PythonOperator(
        task_id='ingestion',
        python_callable=ingestion
    )

    preprocess_task = PythonOperator(
        task_id='preprocessing',
        python_callable=preprocessing
    )

    train_task = PythonOperator(
        task_id='model_training',
        python_callable=model_training
    )

    validate_task = PythonOperator(
        task_id='model_validation',
        python_callable=model_validation
    )

    deploy_task = PythonOperator(
        task_id='model_deployment',
        python_callable=model_deployment
    )

    ingest_task >> preprocess_task >> train_task >> validate_task >> deploy_task
