# -*- coding:utf-8 -*-
# @Copyright 2019 mobvoi Inc.
# @Author   :wbzhu


import jieba


def jieba_wordseg(text):
    """使用结巴分词进行分词

    Keyword arguments:
    text  --  需要处理的文本,type需要是unicode
    """
    word_list = list(jieba.cut(text))
    return word_list