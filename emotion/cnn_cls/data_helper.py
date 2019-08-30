# coding=utf-8

import numpy as np
import re
import os
import pandas as pd
from config import GlobalConfig

import regex

cf = GlobalConfig()

def load_raw_data(filename):
    xls = pd.ExcelFile(filename)
    sheet_name = xls.sheet_names[0]
    sheet1 = xls.parse(sheet_name)

    queries = list(sheet1["query"])
    queries = [x.strip() for x in queries]

    labels = list(sheet1["label"])
    labels = [str(x.strip()) for x in labels]

    return np.array(queries), np.array(labels)


def load_sentiment_dict(filename):
    xls = pd.ExcelFile(filename)
    sheet_name = xls.sheet_names[0]
    sheet1 = xls.parse(sheet_name)

    words = list(sheet1["word"])
    words = [x for x in words]

    labels = list(sheet1["label"])
    labels = [str(x.strip()) for x in labels]

    vocab = {x: y for (x, y)in zip(words, labels)}

    return vocab


def load_vocab(filename):
    vocab = []
    label = []
    # 词
    if os.path.exists(cf.vocab_path):
        print("Using existing files!")
        for line in open(cf.vocab_path):
            line = line.strip().decode('utf-8')
            vocab.append(line)
        vocab = [w.strip() for w in vocab]
        idx2vocab = dict([(x, y) for (y, x) in enumerate(vocab)])
    else:
        vocab = set()
        vocab_list = ['PAD', 'UNK']
        query_raw, _ = load_raw_data(filename)
        length = len(query_raw)
        for i in range(length):
            words = query_raw[i].split()
            for w in words:
                vocab.add(w)
        vocab_list.extend(list(vocab))
        with open(cf.vocab_path, 'w')as f:
            for w in vocab_list:
                f.write(w.encode('utf-8') + "\n")
            f.close()
        idx2vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
        print("create vocab file!")
    # 分类标签
    if os.path.exists(cf.label_path):
        print("Using existing files!")
        with open(cf.label_path, 'r')as f:
            label.extend(f.readlines())
        label_vocab = [w.strip() for w in label]
        idx2label = dict([(x, y) for (y, x) in enumerate(label_vocab)])
    else:
        _, lables_raw = load_raw_data(filename)
        label_vocab = list(set(lables_raw))
        with open(cf.label_path, 'w')as f:
            for w in label_vocab:
                f.write(w.encode('utf-8') + "\n")
            f.close()
        idx2label = dict([(x, y) for (y, x) in enumerate(label_vocab)])
        print("create vocab file!")

    return idx2vocab, vocab, idx2label, label_vocab


def load_pattern(vocab_file):
    def reader(filename):
        fr = []
        for line in open(filename):
            line = line.strip().decode('utf-8')
            fr.append(line)
        res = [w.strip() for w in fr]
        return res

    regex_patterns = {}
    for line in open(vocab_file):
        line = line.strip().decode('utf-8')
        if line:
            name = line[:line.find('=')]
            rt = line[line.find('=') + 1:]
            regex_patterns[name] = rt

    happy = reader(cf.vocab_sentiment_happy)
    angry = reader(cf.vocab_sentiment_angry)
    fear = reader(cf.vocab_sentiment_fear)
    sad = reader(cf.vocab_sentiment_sad)
    surprise = reader(cf.vocab_sentiment_surprise)

    table = {'happy': happy,
             'angry': angry,
             'fear': fear,
             'sad': sad,
             'surprise': surprise}

    _regex_patterns = {}
    for name in regex_patterns:
        rt = regex_patterns[name]
        for tname in table.keys():
            # fill sentiment words
            rt = rt.replace(tname, ('|'.join(table[tname])))
        _regex_patterns[name] = rt

    pattern_vocab = dict([(y, x + 1)
                          for (x, y)in enumerate(regex_patterns.keys())])

    return _regex_patterns, pattern_vocab


def map_file_to_ids(filename):

    def padding(x, max_len):
        res = (np.ones((max_len)) * 0.0).astype(np.int32)
        lx = min(len(x), max_len)
        res[:lx] = x[:lx]
        return res

    def padding_patt(x, max_len):
        res = (np.ones((max_len)) * num_classes).astype(np.int32)
        lx = min(len(x), max_len)
        res[:lx] = x[:lx]
        return res

    print "load raw data..."
    queries, labels = load_raw_data(filename)

    print "load vocab..."
    vocab, _, label_vocab, _ = load_vocab(filename)

    num_classes = len(label_vocab)

    print "load sentiment lexicon..."
    sentiment_dict = load_sentiment_dict(cf.sentiment_lexicon)

    print "map sentiment vector..."
    sentiment_vector = []
    for query in queries:
        words = query.split()
        sents = []
        # 训练数据是分过词的
        for word in words:
            word = word.strip()
            if word in sentiment_dict.keys():
                sents.append(sentiment_dict[word])
        sentiment_vector.append([label_vocab[w]for w in sents])

    sentiment_vector = np.array([padding_patt(x, cf.max_sentiment_len)for x in sentiment_vector])

    print "map word embedding..."
    # 基于词
    # 这个数组里面是词数组
    queries_seg = [x.split()for x in queries]
    # # 词映射成id
    x = np.array([padding([vocab.get(xx, 1)
                           for xx in x], cf.max_sequence_length)for x in queries_seg])
    # 基于字,更准确
    # 这个数组里面是句子
    # queries_seg = [x.replace(' ', '') for x in queries]
    # 字映射成id
    # x = np.array([padding([vocab.get(xx, 1)for xx in x], cf.max_sequence_length)for x in queries_seg])

    y = np.array([label_vocab[w] for w in labels], dtype=np.int32)
    # 正则表达式和相应的索引
    regex_patterns, patterns_vocab = load_pattern(cf.regex_path)

    print "map regex vector..."
    negs = []
    patterns = []
    for i, line in enumerate(queries):
        line = line.replace(' ', '')
        pattern = []
        for name in regex_patterns.keys():
            rt = regex_patterns[name]
            m = regex.search(rt, line)
            if m:
                end = name.find('_')
                pattern_name = name[:end]
                label_id = label_vocab[pattern_name.lower()]
                # if label_id not in pattern:
                pattern.append(label_id)

        patterns.append(pattern)
    patterns = [padding_patt(p, cf.max_pattern_length)for p in patterns]

    # word embedding
    # label
    # 情感词典
    # 正则
    return x, y, sentiment_vector, patterns


def load_embedding(filename, vocab={}):
    fr = open(filename, "r")
    word2vec = {}
    embeddings = []
    # 加载pre-train word2vector
    fr.readline()
    for line in fr.readlines():

        words = line.strip().decode('utf-8').split()
        key = words[0]
        word2vec[key] = [float(x)for x in words[1:]]
    # 针对给定的vocab生成对应的embedding dict
    cnt = 0
    for key, id in vocab.items():
        if key in word2vec.keys():
            embeddings.append(word2vec[key])
            cnt += 1
        else:
            np.random.seed(id)
            embeddings.append(np.random.normal(0, 0.1, cf.embedding_dim))
    return np.array(embeddings, dtype=np.float32)


def Itertool(data, batch_size, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches = int((len(data) - 1) / batch_size + 1)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]
