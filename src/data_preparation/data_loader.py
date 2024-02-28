# src/data_preparation/data_loader.py

import nltk
from nltk.corpus import movie_reviews

def load_movie_reviews():
    nltk.download('movie_reviews')
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    return documents
