# -*- coding:utf-8 -*-
# @Copyright 2019 mobvoi Inc.
# @Author   :wbzhu


from collections import Counter
from sklearn.preprocessing import MinMaxScaler


def location_features(sentences_idx):
    """
    按照句子的位置赋予句子的分数

    Keyword arguments:
    sentences_idx    ----    id
    """
    sentences_idx.sort()
    sentences_size = len(sentences_idx)
    n = float(sentences_size + 2)
    features = {}
    features[sentences_idx[0]] = 2 / n
    for i in sentences_idx[1:-1]:
        features[i] = 1 / n
    features[sentences_idx[-1]] = 2 / n
    return features


def headline_features(headline, sentences, sim_model):
    """
    计算首句与其他句子的分数

    Keyword arguments:
    headline    ----    文章的第一句
    sentences    ----    id to sentence
    sim_model    ----    计算相似度的模型
    """
    features = {}
    for idx in range(len(sentences)):
        score = sim_model.get_similarity(headline, sentences[idx])
        features[idx] = score
    return features

def tfidf_features(sentences, term_idf):
    """

    Keyword arguments:
    """
    score_list = []
    for idx in sentences:
        words = sentences[idx]
        counter = Counter(words)
        score = 0
        for word in counter:
            if not word in term_idf:
                continue
            tf = counter[word]
            score += tf*term_idf[word]
        score_list.append([score])

    sd_scaler = MinMaxScaler()
    sd_scaler.fit(score_list)

    features = {}
    for idx, score in enumerate(sd_scaler.transform(score_list)):
        feat = score[0]
        features[idx] = feat

    return features