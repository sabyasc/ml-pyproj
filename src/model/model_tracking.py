"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Jan 2025
"""
# src libs are fetched from __init__.py
from model.model_evaluation import validation 
from src import pd, mlflow, joblib, json

# Model Tracking is to track the model with model_training outputs. We will follow below steps:
# Step 1: Log the best model, accuracy, precision, recall scores,
# Step 2: Log confusion matrix and classification report,
# Step 3: Save the confusion matrix and classification report to artifacts,
# Step 4: Console log message for successful model tracking,
# Step 5: Return outputs of the model metrics and predictions
def model_tracking():
    outputs = validation()
    mlflow.start_run()
    mlflow.log_param("Best Model", outputs['best_model'])
    mlflow.log_metric("Accuracy", outputs['accuracy'])
    mlflow.log_metric("Precision", outputs['precision'])
    mlflow.log_metric("Recall", outputs['recall'])
    mlflow.log_metric("Overfitting", outputs['overfitting'])
    mlflow.log_metric("F1 Score", outputs['f1_score'])
    mlflow.log_metric("ROC AUC Score", outputs['roc_auc'])
    
    confusion_matrix_path = './confusion_matrix.csv'
    pd.DataFrame(outputs['confusion_matrix']).to_csv(confusion_matrix_path, index=False)
    mlflow.log_artifact(confusion_matrix_path)
    
    classification_report_path = './classification_report.json'
    with open(classification_report_path, 'w') as f:
        json.dump(outputs['classification_report'], f)
    mlflow.log_artifact(classification_report_path)
    
    mlflow.end_run()
    print("================ Model Tracking completed ================")
    return outputs

# Model Testing is to test the model with model_validation outputs. We will follow below steps:
# Step 1: Display first 10 example rows of data with actual and predicted sentiments,
# Step 2: Display number of misclassified and correctly classified samples,
# Step 3: Return outputs of the model metrics and predictions
def model_testing():
    result = validation()
    X_test = result['X_test']
    y_test = result['y_test']
    y_pred = result['y_pred']

    test_results = []
    
    for i in range(10):
        test_results.append({
            "Tweet": X_test[i],
            "Actual Sentiment": y_test[i],
            "Predicted Sentiment": y_pred[i]
        })

    misclassified = [(X_test[i], y_test[i], y_pred[i]) for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    correct_classified = [(X_test[i], y_test[i], y_pred[i]) for i in range(len(y_test)) if y_test[i] == y_pred[i]]

    print(f"Number of misclassified samples: {len(misclassified)}")
    print(f"Number of correctly classified samples: {len(correct_classified)}")

    print("================ Model Testing completed ================")
    return {
        "test_results": test_results,
        "misclassified": misclassified,
        "correct_classified": correct_classified
    }

# Model Deployment is to deploy the model with model_validation outputs. We will follow below steps:
# Step 1: Save the model metadata and trained model to 'metadata.pkl' and 'trained_model.pkl' respectively,
# Step 2: Console log message for successful model deployment,
# Step 3: Return metadata of the model
def model_deployment():
    metadata = validation()
    
    joblib.dump(metadata, './metadata/metadata.pkl')
    print("Model data saved to 'metadata.pkl'")

    ensemble_model = metadata['best_model']
    joblib.dump(ensemble_model, './metadata/trained_model.pkl')
    print("Trained ensemble model saved to 'trained_model.pkl'")
    
    print("================ Model Deployment completed ================")
    return metadata
