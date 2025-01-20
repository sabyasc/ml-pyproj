"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Jan 2025
"""
# src libs are fetched from __init__.py
import nltk
from preprocess.data_ingestion import ingestion
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Required nltk packages 
nltk.download('punkt_tab')
nltk.download('stopwords')

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
