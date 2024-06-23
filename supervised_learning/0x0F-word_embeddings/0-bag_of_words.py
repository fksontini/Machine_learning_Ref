#!/usr/bin/env python3
""" NLP - Word Embeddings """
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ Create a bag of words embedding matrix """
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).todense()
    features = vectorizer.get_feature_names()
    return embeddings, features
