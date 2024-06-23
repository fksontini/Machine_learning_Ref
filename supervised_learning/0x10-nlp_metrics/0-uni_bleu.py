#!/usr/bin/env python3
""" NLP - Evaluation Metrics """
import numpy as np


def uni_bleu(references, sentence):
    """ Calculate the unigram BLEU score for a sentence"""
    c = len(sentence)
    refslen = np.array([len(r) for r in references])
    refminidx = np.argmin(np.abs(refslen - c))
    r = len(references[refminidx])
    if r > c:
        bp = np.exp(1 - r / c)
    else:
        bp = 1
    words = dict()
    for word in sentence:
        for ref in references:
            if word in words:
                if words[word] < ref.count(word):
                    words.update({word: ref.count(word)})
            else:
                words.update({word: ref.count(word)})
    p = sum(words.values())
    return bp * p / c
