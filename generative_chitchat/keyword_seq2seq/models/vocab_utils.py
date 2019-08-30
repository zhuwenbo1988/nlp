#!/usr/bin/env python
# coding: utf-8


import codecs
import re
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops


DEFAULT_SEPARATOR = "  |  "
SEPARATOR_SYMBOL = "|"

UNK, UNK_ID = "<unk>", 0
SOS, SOS_ID = "<s>", 1
EOS, EOS_ID = "</s>", 2

RESERVED_WORDS = [UNK, SOS, EOS]


def load_vocab(vocab_file):
    vocab = []

    if tf.gfile.Exists(vocab_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            vocab_size = 0
            for word in f:
                vocab_size += 1
                vocab.append(word.strip())
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_file)

    return vocab, vocab_size


def create_vocab_dict(vocab_file, start_index=0):
    """Creates vocab tables for vocab_file."""
    if tf.gfile.Exists(vocab_file):

        vocab_dict = defaultdict(lambda: UNK_ID)

        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            for id, word in enumerate(f):
                w = word.strip()
                vocab_dict[w] = id + start_index

        return vocab_dict
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_file)
        

def create_vocab_table(vocab_file):
    """Creates vocab tables for vocab_file."""
    return lookup_ops.index_table_from_file(vocab_file, default_value=UNK_ID)


def create_rev_vocab_table(vocab_file):
    return lookup_ops.index_to_string_table_from_file(vocab_file, default_value=UNK)
        

def initialize_vocabulary(hparams):
    _create_vocabulary(hparams.vocab_file, hparams.topic_vocab_file, hparams.train_data, hparams.vocab_size)

    vocab_table = create_vocab_dict(hparams.vocab_file)
    topic_vocab_table = create_vocab_dict(hparams.topic_vocab_file)

    for w in topic_vocab_table:
        topic_vocab_table[w] = vocab_table[w]

    return vocab_table, topic_vocab_table


def _create_vocabulary(vocab_path, topic_vocab_path, data_path, max_vocabulary_size, normalize_digits=False):
    """A modified version of vocab.create_vocabulary
    """

    if tf.gfile.Exists(vocab_path) and tf.gfile.Exists(topic_vocab_path):
        return

    print("Creating vocabulary files from data %s" % data_path)
    dialog_vocab, topic_vocab = {}, {}

    def normalize(word):
        if normalize_digits:
            if re.match(r'[\-+]?\d+(\.\d+)?', word):
                return '<number>'

        return word

    with codecs.getreader('utf-8')(
            tf.gfile.GFile(data_path, mode="rb")) as f:
        counter = 0
        for line in f:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)

            dialog_line, topic_line = tuple(line.split(DEFAULT_SEPARATOR))
            dialog_tokens, topic_tokens = dialog_line.strip().split(), topic_line.strip().split()

            for word in dialog_tokens:
                word = normalize(word)

                if word in dialog_vocab:
                    dialog_vocab[word] += 1
                else:
                    dialog_vocab[word] = 1

            for word in topic_tokens:
                word = normalize(word)

                if word in topic_vocab:
                    topic_vocab[word] += 1
                else:
                    topic_vocab[word] = 1

    for word in topic_vocab:
        if word in dialog_vocab:
            topic_vocab[word] += dialog_vocab[word]

    topic_vocab_list = sorted(topic_vocab, key=topic_vocab.get, reverse=True)
    with codecs.getwriter('utf-8')(
            tf.gfile.GFile(topic_vocab_path, mode="wb")) as topic_vocab_file:
        for w in topic_vocab_list:
            topic_vocab_file.write(w + "\n")

    for reserved_word in RESERVED_WORDS:
        if reserved_word in dialog_vocab:
            dialog_vocab.pop(reserved_word)

    dialog_vocab_list = RESERVED_WORDS + sorted(dialog_vocab, key=dialog_vocab.get, reverse=True)

    if len(dialog_vocab_list) > max_vocabulary_size:
        dialog_vocab_list = dialog_vocab_list[:max_vocabulary_size]

    for word in topic_vocab:
        if word in dialog_vocab_list and word not in RESERVED_WORDS:
            dialog_vocab_list.remove(word)

    with codecs.getwriter('utf-8')(
            tf.gfile.GFile(vocab_path, mode="wb")) as vocab_file:
        for w in dialog_vocab_list:
            vocab_file.write(w + "\n")

        for w in topic_vocab_list:
            vocab_file.write(w + "\n")

    print("Topic vocabulary with {} words created".format(len(topic_vocab_list)))
    print("Vocabulary with {} words created".format(len(topic_vocab_list)))

    del topic_vocab
    del dialog_vocab
    

def get_translation(ncm_outputs, sent_id):
    """Given batch decoding outputs, select a sentence and turn to text."""
    # Select a sentence
    output = ncm_outputs[sent_id, :].tolist() if len(ncm_outputs.shape) > 1 else ncm_outputs.tolist()

    eos = EOS.encode("utf-8")

    # If there is an eos symbol in outputs, cut them at that point.
    if eos in output:
        output = output[:output.index(eos)]

    return b" ".join(output)

