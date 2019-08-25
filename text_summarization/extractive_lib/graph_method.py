# -*- coding:utf-8 -*-
# @Copyright 2019 mobvoi Inc.
# @Author   :wbzhu


import networkx as nx
import numpy as np


def rank_sentences(sentences, sim_model, pagerank_config={'alpha': 0.85, }):
    """将句子按照关键程度从大到小排序

    Keyword arguments:
    sentences         --  字典, key是句子id, value是句子的单词列表
    sim_model         --  计算两个句子的相似性
    pagerank_config   --  pagerank的设置
    """
    sorted_sentences = {}
    sentences_num = len(sentences)
    graph = np.zeros((sentences_num, sentences_num))

    for x in xrange(sentences_num):
        for y in xrange(x, sentences_num):
            similarity = sim_model.get_similarity(sentences[x], sentences[y])
            graph[x, y] = similarity
            graph[y, x] = similarity

    nx_graph = nx.from_numpy_matrix(graph)
    # scores is a dict
    scores = nx.pagerank(nx_graph, **pagerank_config)
    sorted_scores = sorted(scores.iteritems(), key=lambda x: x[1], reverse=True)

    for index, score in sorted_scores:
        sorted_sentences[index] = score

    return sorted_sentences