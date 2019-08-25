# -*- coding:utf-8 -*-
# @Copyright 2019 mobvoi Inc.
# @Author   :wbzhu


def split_to_short_sentence(doc):
    """将长文本进行断句

    Keyword arguments:
    doc  --  需要处理的长文本,type需要是unicode
    """
    sentences = []
    long_text_list = split_to_long_sentence(doc)
    for long_text in long_text_list:
        short_text_list = long_text.split('SSP')
        sentences.extend(short_text_list)
    return sentences

def split_to_long_sentence(doc):
    """将长文本进行断句

    Keyword arguments:
    doc  --  需要处理的长文本,type需要是unicode
    """
    sentences = doc.split('LSP')
    return sentences