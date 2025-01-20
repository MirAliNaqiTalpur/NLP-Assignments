# Assignment 1: That’s What I LIKE

## Overview
This project focuses on building a Natural Language Processing (NLP) system to identify and retrieve the top 10 contextually similar paragraphs to a given query. It involves implementing, training, and analyzing word embedding models (Word2Vec and GloVe) and deploying the system as a web application.

---

## Features

1. **Word Embedding Implementation**:
   - Modified Word2Vec with and without negative sampling.
   - GloVe embeddings trained from scratch with dynamic window size.
   - Pre-trained GloVe embeddings using Gensim.

2. **Evaluation and Analysis**:
   - Comparison of models on training loss, time, and accuracy.
   - Syntactic and semantic accuracy tests using Word analogy datasets.
   - Spearman’s rank correlation to measure similarity with human-annotated datasets.

3. **Web Application**:
   - Simple Flask-based web app with an input box for user queries.
   - Retrieval of top 10 most similar contexts based on dot product computations.

---

## Directory Structure

```
|-- A1/
|   |-- README.md                         # Project README file
|   |-- st125001_mirali_A1_assignment.ipynb # Jupyter notebook for the assignment
|   |-- word-test.v1.txt                  # Test data for word similarity or embeddings evaluation
|   |-- wordsim_similarity_goldstandard.txt # Word similarity gold standard (for evaluation)
|   |-- app/
|   |   |-- app.py                       # Flask application entry point
|   |   |-- embedding/                   # Contains trained embedding models
|   |   |   |-- glove_embeds.pickle      # GloVe embedding model (Pickle format)
|   |   |   |-- neg_embeds.pickle        # Negative embeddings (Pickle format)
|   |   |   |-- skipgram_embeds.pickle   # Skipgram embeddings (Pickle format)
|   |   |-- templates/                   # HTML templates for the web interface
|   |   |   |-- index.html               # HTML template for the main page
|   |-- glove.6B/                        # GloVe model folder with pre-trained vectors
|   |   |-- glove.6B.100d.txt            # GloVe 100-dimensional pre-trained embeddings
```

---



## Corpus Used for Training

The embeddings were trained on a subset of the [NLTK Reuters Corpus](https://www.nltk.org/book/ch02.html), which contains categorized news articles. Preprocessing steps included:

1. Tokenization and lowercasing.
2. Removal of stopwords and punctuation.
3. Limiting the vocabulary to the top 10,000 most frequent words.

This corpus provides a domain-specific dataset for testing and analyzing word embeddings.


## Results

### Model Accuracies and Training Time

| **Model**          | **Window Size** | **Training Loss** | **Training time** | **Syntactic Accuracy** | **Semantic accuracy** |
|--------------------|:---------------:|:-----------------:|:-----------------:|:----------------------:|:---------------------:|
| **Skipgram**       |        2        |      8.24      |       3.18 s      |          0.        |           0          |
| **Skipgram (NEG)** |        2        |       14.95      |       2.99 s      |           0           |           0         |
| **GloVe**          |        2        |       7.93      |      59.40 s      |           0           |          0.2        |
| **GloVe (Gensim)** |        -        |         -         |         -         |         41.79%         |         90.32        |

### Correlation between Model Dot Product and Human Judgments

| **Model**                | **Skipgram** | **NEG** | **GloVe** | **GloVe (gensim)** |
|--------------------------|--------------|---------|-----------|--------------------|
| **Spearman Correlation** |    0.172    |  -0.032 |   -0.058  |       0.5431       |

---

## Web Application

The web application allows users to input a query and retrieve the top 10 contextually similar paragraphs. The Flask-based app leverages the trained embeddings (Skipgram, GloVe, etc.) to compute similarities between the query and a dataset of paragraphs. The system ranks these paragraphs based on the highest dot product similarity.

### Features:
- Simple input interface for entering queries.
- Retrieval of similar paragraphs based on trained word embeddings.
- Displays the most similar paragraphs to the user’s query.

Here is a screenshot of the web app interface:

![Web app screenshot](https://github.com/MirAliNaqiTalpur/NLP-Assignments/blob/main/A1/webapp_screenshot.png)

You can access the web application through the provided link (once hosted) or run it locally by setting up the Flask environment.

---
## Additional Notes

1. The dataset used for training was derived from publicly available sources.
2. Limited corpus size and training time affected the accuracy of custom embeddings.
3. Pre-trained GloVe embeddings (Gensim) performed significantly better than custom models.
4. Future improvements could involve using a larger corpus and optimizing hyperparameters.

---

## References

1. [Word2Vec Paper (Mikolov et al.)](https://arxiv.org/pdf/1301.3781.pdf)
2. [GloVe Paper (Pennington et al.)](https://aclanthology.org/D14-1162.pdf)
3. [Word Analogies Dataset](https://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt)
4. [Word Similarity Dataset](http://alfonseca.org/eng/research/wordsim353.html)
