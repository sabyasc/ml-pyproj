"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Jan 2025
"""
from preprocess.data_preprocessing import preprocessing
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
    
# Data Training is to train the model with data_preprocessing data. We will follow below steps:
# Step 1: Perform sentiment analysis on 'Tweet_Text', split data into X and y
# Step 2: Split data into training and testing sets to be passed in ML model, 
# Step 4: Vectorize the data using TfidfVectorizer,
# Step 5: Train the model using ensemble of LogisticRegression, RandomForest, SVM, etc,
# Step 6: Predict the model using X_test,
# Step 7: Calculate accuracy, precision, recall scores, and best model,
# Step 8: Return the model metrics and predictions
def train():
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
