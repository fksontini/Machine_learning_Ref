#!/usr/bin/env python3
""" NLP - Word Embeddings """
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """ Convert a gensim word2vec model to a keras Embedding layer """
    return model.wv.get_keras_embedding(train_embeddings=False)
