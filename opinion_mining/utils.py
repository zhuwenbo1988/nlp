import internal_config
from gensim.models import word2vec
from database import mongodb_client
from pymongo import DeleteOne
import requests


POLARITY_UNK = internal_config.polarity_unk
PU_KEEP = internal_config.keep_p
PU_pattern = internal_config.PU_pattern
PU = internal_config.PU
WORD2VEC_URL = internal_config.WORD2VEC_URL


class Resources_data:
    def __init__(self, *args):
        pass


def load_resources(domain):
    """"""
    resources_data = Resources_data()
    """加载分词，词性，依存"""
    documents = mongodb_client.db['comments'].find({'domain': domain})
    seg_list = []
    pos_list = []
    """这里doc['seg']是string，下面还要按空格分隔"""
    for idx, doc in enumerate(documents):
        seg_list.append(doc['seg'].split(' '))
        pos_list.append(doc['pos'].split(' '))

    """加载按标点符号分隔的分词，词性"""
    seg_pu_list, pos_pu_list = _split_by_pu(seg_list, pos_list)

    """加载第三方情感词典"""
    general_opinion_doc = mongodb_client.db['opinion_resources'].find_one({'doc_type': 'general_opinion'})
    general_opinion = general_opinion_doc['lexicon']

    """加载停用词表"""
    stopwords = []
    stopwords_docs = mongodb_client.db['opinion_resources'].find({'doc_type': 'stopwords'})
    for doc in stopwords_docs:
        for word in doc['lexicon']:
            stopwords.append(word)

    """加载用户定义词典"""
    user_defined_aspect = []
    udf_doc = mongodb_client.db['opinion_resources'].find_one({'doc_type': 'product_tag', 'domain': domain})
    for word in udf_doc['lexicon']:
        user_defined_aspect.append(word)
    """加载word2vec模型"""
    word2vec_model = load_word2vec_model()

    resources_data.seg_list = seg_list
    resources_data.seg_pu_list = seg_pu_list
    resources_data.pos_list = pos_list
    resources_data.pos_pu_list = pos_pu_list
    resources_data.general_opinion = general_opinion
    resources_data.stopwords = stopwords
    resources_data.user_defined_aspect = user_defined_aspect
    resources_data.word2vec_model = word2vec_model

    return resources_data


def _split_by_pu(seg_list, pos_list):
    """将每条评论的分词和词性标注结果按照标点符号分割，每行是一个短句，重新写入一个文件"""
    seg_pu_list = []
    pos_pu_list = []
    for x, word_line in enumerate(seg_list):
        start_idx = 0
        no_error = True
        for y, word in enumerate(word_line):
            if word in PU_KEEP and pos_list[x][y] in PU:
                end_idx = y
                seg_pu_list.append(word_line[start_idx: end_idx])
                pos_pu_list.append(pos_list[x][start_idx: end_idx])
                start_idx = y + 1
            elif word in PU_KEEP and pos_list[x][y] not in PU:
                no_error = False
            else:
                continue
        else:
            if start_idx != len(word_line) and no_error:
                seg_pu_list.append(word_line[start_idx:])
                pos_pu_list.append(pos_list[x][start_idx:])
    return seg_pu_list, pos_pu_list


def load_word2vec_model():
    r = requests.get(WORD2VEC_URL)
    with open('word2vec.model', 'wb') as f:
        f.write(r.content)
    model = word2vec.Word2Vec.load("word2vec.model")
    return model

