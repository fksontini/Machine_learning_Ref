#!/usr/bin/env python3
""" NLP - Evaluation Metrics """
import numpy as np


def ngrams(sentences, n):
    """ ngrams sentence """
    ngrams_sentence = []
    for i in range(len(sentences) - n + 1):
        ngrams_sentence.append(' '.join(sentences[i:i + n]))
    return ngrams_sentence


def ngram_bleu(references, sentence, n):
    """ Calculate the n-gram BLEU score for a sentence """
    c = len(sentence)
    refslen = np.array([len(r) for r in references])
    refminidx = np.argmin(np.abs(refslen - c))
    r = len(references[refminidx])
    if r > c:
        bp = np.exp(1 - r / c)
    else:
        bp = 1
    s = ngrams(sentence, n)
    r = list(ngrams(ref, n) for ref in references)
    words = dict()
    for word in s:
        for ref in r:
            if word in words:
                if words[word] < ref.count(word):
                    words.update({word: ref.count(word)})
            else:
                words.update({word: ref.count(word)})
    p = sum(words.values())
    return bp * p / len(s)
