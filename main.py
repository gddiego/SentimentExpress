# main.py

from src.models.sentiment_classifier import extract_features, train_classifier
from src.data_preparation.data_loader import load_movie_reviews  # Importação correta

def main():
    documents = load_movie_reviews()  # Corrigido o uso da função
    featuresets = extract_features(documents)
    train_set, test_set = featuresets[100:], featuresets[:100]
    classifier = train_classifier(train_set)
    acc = accuracy(classifier, test_set)
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()
