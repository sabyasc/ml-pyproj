"""
author: sabyasc
github: https://github.com/sabyasc
created: Dec 2024
"""
import pandas as pd
import nltk, mlflow, mlflow.sklearn
import joblib, json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,\
    confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Required nltk packages 
nltk.download('punkt_tab')
nltk.download('stopwords')

# Data Ingestion is to fetech data from sources, We will follow below steps:
# Step 1: Read data from source (csv, json, db, APIs, etc),
# Step 2: Display first 10 example rows of data to understand the data,
# Step 3: Return the data as Dataframe to next step for processing
def ingestion():
    data = pd.read_csv(r'C:\GitHub\ml-pyproj\ml-pipelines\dataset\train.csv')
    df = pd.DataFrame(data)
    count = df.head(50)
    print("================ Data Ingestion completed ================")
    return count

# Data Preprocessing is to clean the dataframe received from data_ingestion. We will follow below steps:
# Step 1: Remove speacial chars, convert to lowercase, tokenization (breaking into seperate words), 
# Step 2: Remove stopwords, remove short words, join tokens to string, remove duplicates, reset index, 
# Step 3: Remove rows with empty 'Tweet_Text', normalization (converting into standard format for system), 
# Step 4: Remove URLs, remove mentions, remove hashtags, remove numbers, remove extra spaces, 
# Step 5: lemmatization (optional - converting words to base form)
def preprocessing():
    df = ingestion().dropna(axis=1, how='any')

    df['Tweet_Text'] = df['Tweet_Text'].str.replace('[^a-zA-Z0-9\s]', '', regex=True).str.lower()
    
    df['Tweet_Text'] = df['Tweet_Text'].apply(word_tokenize)

    sw = set(stopwords.words('english'))
    df['Tweet_Text'] = df['Tweet_Text'].apply(lambda x: [item for item in x if item not in sw])
    
    df['Tweet_Text'] = df['Tweet_Text'].apply(lambda x: [item for item in x if len(item) > 2])
    df['Tweet_Text'] = df['Tweet_Text'].apply(lambda x: ' '.join(x))

    df = df.drop_duplicates()

    df = df.reset_index(drop=True)
    
    df = df[df['Tweet_Text'].str.strip().astype(bool)]

    df['Tweet_Text'] = df['Tweet_Text'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    df['Tweet_Text'] = df['Tweet_Text'].str.replace(r'http\S+|www.\S+', '', regex=True)
    df['Tweet_Text'] = df['Tweet_Text'].str.replace(r'@\w+', '', regex=True)
    df['Tweet_Text'] = df['Tweet_Text'].str.replace(r'#\w+', '', regex=True) 
    df['Tweet_Text'] = df['Tweet_Text'].str.replace(r'\d+', '', regex=True)
    df['Tweet_Text'] = df['Tweet_Text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    lemmatizer = nltk.WordNetLemmatizer()
    nltk.download('wordnet')
    df['Tweet_Text'] = df['Tweet_Text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    
    def get_sentiment(text):
        analysis = TextBlob(text)
        return 1 if analysis.sentiment.polarity > 0 else 0
    
    df['Label'] = df['Tweet_Text'].apply(get_sentiment)
    
    print("================ Data Preprocessing completed ================")
    return df

# Data Training is to train the model with preprocessed data. We will follow below steps:
# Step 1: Perform sentiment analysis on 'Tweet_Text', split data into X and y
# Step 2: Split data into training and testing sets to be passed in ML model, 
# Step 4: Vectorize the data using TfidfVectorizer,
# Step 5: Train the model using ensemble of LogisticRegression, RandomForest, SVM, etc,
# Step 6: Predict the model using X_test,
# Step 7: Calculate accuracy, precision, recall scores, and best model,
# Step 8: Return the model metrics and predictions
def model_training():
    df = preprocessing()
    
    def sentiment_analysis(data):
        analysis = TextBlob(data)
        if analysis.sentiment.polarity > 0:
            return 1 
        else: 
            return 0

    df['Label'] = df['Tweet_Text'].apply(sentiment_analysis)
    X = df['Tweet_Text']
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    regression_model = LogisticRegression()
    randomforest_model = RandomForestClassifier()
    svm_model = SVC(probability=True)
    
    ensemble_model = VotingClassifier(estimators=[
        ('lr', regression_model), 
        ('rf', randomforest_model), 
        ('svm', svm_model)
    ], voting='soft')

    regression_model.fit(X_train_tfidf, y_train)
    randomforest_model.fit(X_train_tfidf, y_train)
    svm_model.fit(X_train_tfidf, y_train)
    ensemble_model.fit(X_train_tfidf, y_train)

    y_pred = ensemble_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    best_model = max(
        [('Logistic Regression', regression_model), 
         ('Random Forest', randomforest_model), 
         ('Support Vector Machine', svm_model)], 
        key=lambda model: model[1].score(X_test_tfidf, y_test)
    )[0]
    
    print("================ Model Training completed ================")

    return {
        "best_model": best_model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }

# Model Evaluation is to evaluate the model with model_training outputs. We will follow below steps:
# Step 1: Calculate f1 score, confusion matrix, classification report,
# Step 2: Display outputs of the model metrics and predictions,
# Step 3: Return outputs of the model metrics and predictions
def model_evaluation():
    outputs = model_training()
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

# Model Validation is to validate the model with model_evaluation outputs. We will follow below steps:
# Step 1: Calculate ROC AUC score, check for overfitting,
# Step 2: Display outputs of the model metrics and predictions,
# Step 3: Return outputs of the model metrics and predictions,
# Step 4: Check for overfitting from train and validation scores,
# Step 5: Return outputs of the model metrics and predictions
def model_validation():
    outputs = model_evaluation()
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

# Model Tracking is to track the model with model_training outputs. We will follow below steps:
# Step 1: Log the best model, accuracy, precision, recall scores,
# Step 2: Log confusion matrix and classification report,
# Step 3: Save the confusion matrix and classification report to artifacts,
# Step 4: Console log message for successful model tracking,
# Step 5: Return outputs of the model metrics and predictions
def model_tracking():
    outputs = model_validation()
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
    result = model_validation()
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
    metadata = model_validation()
    
    joblib.dump(metadata, './metadata/metadata.pkl')
    print("Model data saved to 'metadata.pkl'")

    ensemble_model = metadata['best_model']
    joblib.dump(ensemble_model, './metadata/trained_model.pkl')
    print("Trained ensemble model saved to 'trained_model.pkl'")
    
    print("================ Model Deployment completed ================")
    return metadata

