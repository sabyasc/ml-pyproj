import nltk
import os

# Dynamically determine the download directory relative to the project root
nltk_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "nltk_libs"))
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)