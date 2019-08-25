# -*- coding:utf-8 -*-
# @Copyright 2019 mobvoi Inc.
# @Author   :wbzhu


import math
from collections import defaultdict
import numpy as np


class TFModel():
    def __init__(self, vocab):
        # vocab is list
        self.term_vocab = vocab

    def get_represemtation(self, segments):
        words = list(set(self.term_vocab))
        vector = [float(segments.count(word)) for word in words]
        return vector

    def get_similarity(self, segments_a, segments_b):
        vector_a = self.get_represemtation(segments_a)
        vector_b = self.get_represemtation(segments_b)

        tmp = [vector_a[x] * vector_b[x] for x in xrange(len(vector_a))]
        co_occur_vector = [1 for num in tmp if num > 0.]
        co_occur_num = sum(co_occur_vector)

        if abs(co_occur_num) <= 1e-12:
            return 0.

        denominator = math.log(float(len(segments_a))) + math.log(float(len(segments_b)))

        if abs(denominator) < 1e-12:
            return 0.

        return co_occur_num / denominator

class TFIDFModel():
    def __init__(self, vocab):
        # vocab is list
        self.term_vocab = vocab

    def compute_idf(self, corpus):
        doc_num = len(corpus)

        df = defaultdict(lambda: 0)
        for idx in corpus:
            words = corpus[idx]
            if not words:
                continue
            unique_words = {}.fromkeys(words).keys()
            for word in unique_words:
                df[word] += 1

        self.term_idf = {}
        for word in self.term_vocab:
            self.term_idf[word] = math.log(float(doc_num) / float((df[word] + 1)))

    def get_represemtation(self, segments):
        words = list(set(self.term_vocab))
        vector = [float(segments.count(word)*self.term_idf[word]) for word in words]
        return vector

    def get_similarity(self, segments_a, segments_b):
        vector_a = self.get_represemtation(segments_a)
        vector_b = self.get_represemtation(segments_b)

        vector1 = np.array(vector_a)
        vector2 = np.array(vector_b)

        fenmu = np.linalg.norm(vector1) * (np.linalg.norm(vector2))
        if fenmu == 0:
            return 0
        sim = np.dot(vector1, vector2) / fenmu

        return sim
