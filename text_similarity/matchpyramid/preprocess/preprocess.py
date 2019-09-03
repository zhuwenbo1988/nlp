# -*- coding: utf-8 -*-

import sys
import six
import numpy as np
from tqdm import tqdm
from preparation import *


class Preprocess(object):

    def __init__(self,
                 word_seg_config={},
                 doc_filter_config={},
                 word_filter_config={},
                 word_index_config={}
                 ):
        # set default configuration
        self._word_seg_config = {'enable': True, 'lang': 'cn'}
        # 句子长度过滤
        self._doc_filter_config = {'enable': True, 'min_len': 0, 'max_len': six.MAXSIZE}
        # 去掉停用词  'stop_words': nltk_stopwords.words('english'),
        self._word_filter_config = {'enable': True, 'stop_words': list(), 'min_freq': 20,
                                    'max_freq': six.MAXSIZE, 'words_useless': None}
        self._word_index_config = {'word_dict': None}
        self._word_seg_config.update(word_seg_config)
        self._doc_filter_config.update(doc_filter_config)
        self._word_filter_config.update(word_filter_config)
        self._word_index_config.update(word_index_config)
        self._word_dict = self._word_index_config['word_dict']
        self._words_stats = dict()

    def run(self, file_path):
        print('load...')
        dids, docs = Preprocess.load(file_path)

        if self._word_seg_config['enable']:
            print('word_seg...')
            docs = Preprocess.word_seg(docs, self._word_seg_config)

        if self._doc_filter_config['enable']:
            print('doc_filter...')
            dids, docs = Preprocess.doc_filter(dids, docs, self._doc_filter_config)

        # cf词频，df为出现某个词的文档个数，idf，
        self._words_stats = Preprocess.cal_words_stat(docs)
        if self._word_filter_config['enable']:
            print('word_filter...')
            docs, self._words_useless = Preprocess.word_filter(docs, self._word_filter_config, self._words_stats)
        print('word_index...')
        docs, self._word_dict = Preprocess.word_index(docs, self._word_index_config)
        return dids, docs, self._word_dict

    @staticmethod
    def parse(line):
        subs = line.split(' ', 1)
        if 1 == len(subs):
            return subs[0], ''
        else:
            return subs[0], subs[1]

    @staticmethod
    def load(file_path):
        dids = list()
        docs = list()
        f = codecs.open(file_path, 'r', encoding='utf8')
        for line in tqdm(f):
            line = line.strip()
            if '' != line:
                did, doc = Preprocess.parse(line)
                dids.append(did)
                docs.append(doc)
        f.close()
        return dids, docs

    @staticmethod
    def word_seg_cn(docs):
        docs = [sent.split() for sent in tqdm(docs)]
        return docs

    @staticmethod
    def word_seg(docs, config):
        # 当前函数函数名_en(调用分词函数）
        docs = getattr(Preprocess, '%s_%s' % (sys._getframe().f_code.co_name, config['lang']))(docs)
        return docs

    @staticmethod
    def cal_words_stat(docs):
        words_stats = {}
        docs_num = len(docs)
        for ws in docs:
            for w in ws:
                if w not in words_stats:
                    words_stats[w] = {}
                    words_stats[w]['cf'] = 0
                    words_stats[w]['df'] = 0
                    words_stats[w]['idf'] = 0
                words_stats[w]['cf'] += 1
            for w in set(ws):
                words_stats[w]['df'] += 1
        for w, winfo in words_stats.items():
            words_stats[w]['idf'] = np.log((1. + docs_num) / (1. + winfo['df']))
        return words_stats

    @staticmethod
    def word_filter(docs, config, words_stats):
        if config['words_useless'] is None:
            config['words_useless'] = set()
            # filter with stop_words
            config['words_useless'].update(config['stop_words'])
            # filter with min_freq and max_freq
            for w, winfo in words_stats.items():
                # filter too frequent words or rare words
                if winfo['cf'] < config['min_freq'] or winfo['cf'] > config['max_freq']:
                    config['words_useless'].add(w)
        # filter with useless words
        docs = [[w for w in ws if w not in config['words_useless']] for ws in tqdm(docs)]
        return docs, config['words_useless']

    @staticmethod
    def doc_filter(dids, docs, config):
        new_docs = list()
        new_dids = list()
        for i in tqdm(range(len(docs))):
            if config['min_len'] <= len(docs[i]) <= config['max_len']:
                new_docs.append(docs[i])
                new_dids.append(dids[i])
        return new_dids, new_docs

    @staticmethod
    def build_word_dict(docs):
        word_dict = dict()
        for ws in docs:
            for w in ws:
                word_dict.setdefault(w, len(word_dict))
        return word_dict

    @staticmethod
    def word_index(docs, config):
        if config['word_dict'] is None:
            config['word_dict'] = Preprocess.build_word_dict(docs)
        docs = [[config['word_dict'][w] for w in ws if w in config['word_dict']] for ws in tqdm(docs)]
        return docs, config['word_dict']

    @staticmethod
    def save_lines(file_path, lines):
        f = codecs.open(file_path, 'w', encoding='utf8')
        for line in lines:
            line = line.strip()
            f.write(line + "\n")
        f.close()

    @staticmethod
    def save_dict(file_path, dic, sort=False):
        if sort:
            dic = sorted(dic.items(), key=lambda d:d[1], reverse=False)
            lines = ['%s %s' % (k, v) for k, v in dic]
        else:
            lines = ['%s %s' % (k, v) for k, v in dic.items()]
        Preprocess.save_lines(file_path, lines)

    def save_word_dict(self, word_dict_fp, sort=False):
        Preprocess.save_dict(word_dict_fp, self._word_dict, sort)

    def save_id2doc(self, path, dids, docs):
        fout = open(path, 'w')
        for inum, did in enumerate(dids):
            fout.write('%s %s %s\n' % (did, len(docs[inum]), ' '.join(map(str, docs[inum]))))
        fout.close()