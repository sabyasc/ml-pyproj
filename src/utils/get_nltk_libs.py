"""
author: @sabyasc
github: https://github.com/sabyasc/ml-pyproj
created: Feb 2025
"""
import nltk, os

nltk_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "nltk_libs"))

nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)