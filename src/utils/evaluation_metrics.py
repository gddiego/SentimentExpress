# src/utils/evaluation_metrics.py

import nltk

def accuracy(classifier, test_set):
    return nltk.classify.accuracy(classifier, test_set)
