# -*- coding:utf-8 -*-
# @Copyright 2019 mobvoi Inc.
# @Author   :wbzhu


def gen_fix_length_summary(sentences, summary_length):
    """将句子按照关键程度从大到小排序

    Keyword arguments:
    sentences         --  [(index, raw_text, [(name, score, weight), ...]), ...]
    summary_length  --  摘要的长度,目前为中文字符的个数
    """
    tmp = []
    for idx, text, feats in sentences:
        sentence_score = 0
        for fname, fscore, fweight in feats:
            sentence_score += fweight*fscore
        tmp.append((idx, text, sentence_score))

    summary = {}
    final_score = []
    size = 0
    for idx, text, score in sorted(tmp, key=lambda x: x[2], reverse=True):
        final_score.append((idx, score))
        if text in summary:
            continue
        if size < summary_length:
            summary[text] = idx
            size += len(text)

    summary_text = ','.join([part for part, idx in sorted(summary.iteritems(), key=lambda x: x[1])])

    return summary_text, final_score
