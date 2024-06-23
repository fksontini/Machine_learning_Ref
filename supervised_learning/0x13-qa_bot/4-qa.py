#!/usr/bin/env python3
""" QA Bot """
import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
import os


def question_answer(coprus_path):
    """ Answer questions from multiple reference texts"""
    url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
    embed = hub.load(url)
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    exit_list = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        print('Q:', end='')
        question = input()
        if question.lower() in exit_list:
            print('A: Goodbye')
            break
        sentence_search = [question]
        documents = []
        for file in os.listdir(coprus_path):
            if file.endswith(".md"):
                filename = os.path.join(coprus_path, file)
                with open(filename) as f:
                    doc = f.read()
                    documents.append(doc)
        sentence_search_encoded = embed(sentence_search).numpy()
        embed_encoded = embed(documents).numpy()
        corr = np.dot(sentence_search_encoded, embed_encoded.T)
        argmax = np.argmax(corr)

        questiontk = tokenizer.tokenize(question)
        referencetk = tokenizer.tokenize(documents[argmax])
        tokens = ['[CLS]'] + questiontk + ['[SEP]'] + referencetk + ['[SEP]']

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        type_ids = [0] * (len(questiontk) + 2) + [1] * (len(referencetk) + 1)
        input_ids, input_mask, type_ids = map(
            lambda x:
                tf.expand_dims(tf.convert_to_tensor(x, dtype=tf.int32), 0),
                (input_ids, input_mask, type_ids)
        )
        outputs = model([input_ids, input_mask, type_ids])
        short_start = tf.argmax(outputs[0][0][1:]) + 1
        short_end = tf.argmax(outputs[1][0][1:]) + 1

        answer_tokens = tokens[short_start: short_end + 1]
        if not answer_tokens:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A:' + tokenizer.convert_tokens_to_string(answer_tokens))