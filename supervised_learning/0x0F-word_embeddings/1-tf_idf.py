#!/usr/bin/env python3
""" NLP - Word Embeddings """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ Create a TF-IDF embedding """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).todense()
    features = vectorizer.get_feature_names()
    return embeddings, features
