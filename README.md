
# Projeto para estudos

Projeto visa aprendizagem de maquina com python.


## Roadmap

- Aprender a utilizar python para aprendizagem natural IA

- Aprender a tratar e adicionar parametros generalista para criação de uma inteligencia com aprendizagem natural.


## Instalação

Instale my-project com npm

```bash
  python -m pip install nltk
  python main.py
```
    
## Main.py

```python
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

```


## Funcionalidades

- Analise filmes indicando ser negativo ou positivo(bom ou ruim)
- Preview em tempo real


## Autores

- [@gddiego](https://github.com/gddiego)


## Referência

 - [Aprender python](https://learnxinyminutes.com/docs/python/)
 - [Documentação python](https://www.python.org/)

