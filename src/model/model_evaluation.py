"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Jan 2025
"""
from model.model_training import train
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score

# Model Evaluation is to evaluate the model with model_training outputs. We will follow below steps:
# Step 1: Calculate f1 score, confusion matrix, classification report,
# Step 2: Display outputs of the model metrics and predictions,
# Step 3: Return outputs of the model metrics and predictions
def evaluation():
    outputs = train()
    y_test = outputs['y_test']
    y_pred = outputs['y_pred']

    f1 = f1_score(y_test, y_pred)
    c_matrix = confusion_matrix(y_test, y_pred)
    cl_report = classification_report(y_test, y_pred)

    outputs.update({
        "f1_score": f1,
        "confusion_matrix": c_matrix.tolist(),
        "classification_report": cl_report
    })

    report_dict = {}
    for line in cl_report.split('\n'):
        if line.strip():
            row = line.split()
            if len(row) == 5:
                report_dict[row[0]] = {
                    "precision": float(row[1]),
                    "recall": float(row[2]),
                    "f1-score": float(row[3]),
                    "support": int(row[4])
                }
            elif len(row) == 6:
                report_dict[row[0] + ' ' + row[1]] = {
                    "precision": float(row[2]),
                    "recall": float(row[3]),
                    "f1-score": float(row[4]),
                    "support": int(row[5])
                }
            elif row[0] == 'accuracy':
                report_dict['accuracy'] = {
                    "precision": "",
                    "recall": "",
                    "f1-score": float(row[1]),
                    "support": int(row[2])
                }

    print(f"Preferred Model: {outputs['best_model']}")
    print(f"Accuracy: {outputs['accuracy']}")
    print(f"Precision: {outputs['precision']}")
    print(f"Recall: {outputs['recall']}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{c_matrix}")
    print(f"Classification Report:\n{cl_report}")
    
    outputs["classification_report"] = report_dict

    print("================ Model Evaluation completed ================")
    return outputs

# Model Validation is to validate the model with model evaluation outputs. We will follow below steps:
# Step 1: Calculate ROC AUC score, check for overfitting,
# Step 2: Display outputs of the model metrics and predictions,
# Step 3: Return outputs of the model metrics and predictions,
# Step 4: Check for overfitting from train and validation scores,
# Step 5: Return outputs of the model metrics and predictions
def validation():
    outputs = evaluation()
    accuracy = outputs['accuracy']
    y_test = outputs['y_test']
    y_pred = outputs['y_pred']

    roc_auc = roc_auc_score(y_test, y_pred)
    outputs['roc_auc'] = roc_auc

    train_score = outputs['accuracy']
    validation_score = accuracy
    overfitting = train_score > validation_score + 0.05
    outputs['overfitting'] = overfitting

    print(f"ROC AUC Score: {roc_auc}")
    print(f"Overfitting: {overfitting}")

    print("================ Model Validation completed ================")
    return outputs
