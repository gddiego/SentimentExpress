# src/models/sentiment_classifier.py

import nltk
from src.data_preparation.data_loader import load_movie_reviews

def extract_features(documents):
    documents = load_movie_reviews()
    all_words = nltk.FreqDist(w.lower() for w in documents.words())
    word_features = list(all_words)[:2000]

    def document_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in document_words)
        return features

    featuresets = [(document_features(d), c) for (d, c) in documents]
    return featuresets

def train_classifier(train_set):
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    return classifier
