# -*- coding:utf-8 -*-
# @Copyright 2019 mobvoi Inc.
# @Author   :wbzhu


import ConfigParser
import json
from collections import Counter, defaultdict
from preprocess import process_cn as cn_pp
from preprocess import wordseg_cn as cn_seg
from preprocess.filter_cn import FilterCN
import split_doc as spliter
from utils.text_representation import TFModel, TFIDFModel
from extractive_lib import graph_method
from extractive_lib import features_for_extractive
from summary import feature_based as summary_generator
import jieba
from metric import eval

# 配置文件
config = ConfigParser.ConfigParser()
config.read("config.conf")

ssp = config.get("preprocess", "special_stop_punctuation")
split_tag = []
if ssp:
    split_tag = ssp.split(',')

# 文本过滤器
text_filter = FilterCN(config.get("preprocess", "stopwords_location"), max_langth=config.get("preprocess", "max_text_length"))


def aggregate_features(all, feature_name, feature_weight, idx2score):
    """
    聚合所有特征

    :param all:
    :param feature_name:
    :param feature_weight:
    :param idx2score:
    :return:
    """
    result = []
    for idx, text, feats in all:
        if idx in idx2score:
            feats.append((feature_name, idx2score[idx], feature_weight))
        result.append((idx, text, feats))
    return result


def preprocess_sentence(sent):
    """
    预处理句子

    :param sent:
    :return:
    """
    # 特殊字符监测
    sent = cn_pp.detect_enword(sent)
    sent = cn_pp.detect_datestr(sent)
    sent = cn_pp.detect_number(sent)
    sent = cn_pp.detect_bracket_content(sent)
    sent = cn_pp.detect_nonstop_punctuation(sent, config.get("preprocess", "cn_nonstop_punctuation").decode('utf-8'))
    sent = cn_pp.detect_nonstop_punctuation(sent, config.get("preprocess", "en_nonstop_punctuation").decode('utf-8'))
    # 分词
    words = cn_seg.jieba_wordseg(sent.strip())
    # 过滤
    words = text_filter.filter(words)
    return words


def preprocess_doc(doc):
    # 繁体,全角
    prep_doc = cn_pp.convert_tcn2scn(doc)
    prep_doc = cn_pp.convert_strQ2B(prep_doc)

    # 检测断句符号
    prep_doc = cn_pp.detect_long_stop_punctuation(prep_doc, list(config.get("preprocess", "cn_long_stop_punctuation").decode('utf-8')))
    prep_doc = cn_pp.detect_long_stop_punctuation(prep_doc, list(config.get("preprocess", "en_long_stop_punctuation").decode('utf-8')))
    prep_doc = cn_pp.detect_short_stop_punctuation(prep_doc, list(config.get("preprocess", "cn_short_stop_punctuation").decode('utf-8')))
    prep_doc = cn_pp.detect_short_stop_punctuation(prep_doc, list(config.get("preprocess", "en_short_stop_punctuation").decode('utf-8')))
    prep_doc = cn_pp.detect_short_stop_punctuation(prep_doc, split_tag)

    return prep_doc

def process(doc, summary_length):
    """
    处理文章,生成摘要

    :param doc:
    :param summary_length:
    :return:
    """
    input_doc = doc.decode('utf-8')
    input_doc = preprocess_doc(input_doc)

    # 分割长句
    cluster_list = spliter.split_to_long_sentence(input_doc)

    # 处理第一句
    headline = cluster_list[0]
    headline_words = preprocess_sentence(headline)
    headline_vocab = {}.fromkeys(headline_words).keys()

    # 分割短句
    paragraph_list = spliter.split_to_short_sentence(input_doc)

    text_rank_input = {}
    doc_all_words = []
    all_feats = []
    for idx, text in enumerate(paragraph_list):
        # 所有特征
        all_feats.append((idx, text.strip(), []))
        words = preprocess_sentence(text)
        text_rank_input[idx] = words
        doc_all_words.extend(words)

    doc_vocab = {}.fromkeys(doc_all_words).keys()
    doc_tf_vocab = Counter(doc_all_words)

    tf_vector_model = TFModel(doc_vocab)

    # tfdf_vector_model = TFIDFModel(headline_vocab)
    tfdf_vector_model = TFIDFModel(doc_tf_vocab)
    tfdf_vector_model.compute_idf(text_rank_input)

    # text rank 特征
    text_rank_result = graph_method.rank_sentences(text_rank_input, tf_vector_model)
    all_feats = aggregate_features(all_feats, 'Text_Rank', 1.0, text_rank_result)

    # 位置特征
    location_result = features_for_extractive.location_features(text_rank_input.keys())
    all_feats = aggregate_features(all_feats, 'location', 1.0, location_result)

    # tfidf特征
    tfidf_result = features_for_extractive.tfidf_features(text_rank_input, tfdf_vector_model.term_idf)
    all_feats = aggregate_features(all_feats, 'tfidf', 1.0, tfidf_result)

    # 与首句的相似度特征
    headline_result = features_for_extractive.headline_features(headline_words, text_rank_input, tfdf_vector_model)
    all_feats = aggregate_features(all_feats, 'headline', 1.0, headline_result)

    # 用所有特征对句子进行排序
    summary, final_score = summary_generator.gen_fix_length_summary(all_feats, summary_length)

    return summary


def compress_by_ratio(doc, compress_ratio=0.1):
    summary_length = len(doc.decode('utf-8')) * compress_ratio
    return process(doc, summary_length)


def compress_by_fix_length(doc, compress_length=60):
    return process(doc, compress_length)


if __name__ == '__main__':
    dir = 'data/financial_data/test_1000_data.txt'
    score_all = defaultdict(list)
    for doc_id, line in enumerate(open(dir)):
        JSON = json.loads(line.strip())
        doc = JSON['content']

        summary = compress_by_fix_length(doc.encode('utf-8'))

        title = JSON['title']
        cands = ' '.join(jieba.cut(title))
        cands = {'generated_description{}'.format(doc_id): cands.strip()}
        refs = ' '.join(jieba.cut(summary))
        refs = {'generated_description{}'.format(doc_id): [refs.strip()]}
        bleu_evaluator = eval.Evaluate()
        blue_value = bleu_evaluator.evaluate(live=True, cand=cands, ref=refs)
        score_all['Bleu_1'].append(blue_value['Bleu_1'])
        score_all['Bleu_2'].append(blue_value['Bleu_2'])
        score_all['Bleu_3'].append(blue_value['Bleu_3'])
        score_all['Bleu_4'].append(blue_value['Bleu_4'])
    for key in score_all:
        average_score = sum(score_all[key])/len(score_all)
        print '{}:{}'.format(key, average_score)

