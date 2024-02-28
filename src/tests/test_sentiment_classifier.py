from src.models.sentiment_classifier import extract_features, train_classifier
from src.utils.evaluation_metrics import accuracy

def test_sentiment_analysis():
    documents = load_movie_reviews()
    featuresets = extract_features(documents)
    train_set, test_set = featuresets[100:], featuresets[:100]
    classifier = train_classifier(train_set)
    assert accuracy(classifier, test_set) > 0.7