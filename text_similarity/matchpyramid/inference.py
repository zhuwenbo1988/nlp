# -*- coding:utf-8 -*-

import os
import sys
import json
import codecs
import numpy as np
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')

def load_word_dict(filename):
    """加载字典"""
    word_dict = {}
    for line in open(filename):
        lines = line.strip().split()
        if len(lines) != 2:
            continue
        word = lines[0].decode("utf-8")
        wid = int(lines[1])
        word_dict[word] = wid
    print '[%s]\n\tWord dict size: %d' % (filename, len(word_dict))
    return word_dict

def load_dataset(filename, dataset_name):
    """加载测试数据"""
    dataset_list = []
    with codecs.open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            candidates = json.loads(line)
            query = ""
            pairs = []
            for key, value in candidates.items():
                if dataset_name == "weibo" and key == "query":
                    query = value
                    continue
                if key == "pairs":
                    pairs = value
                    continue
            for pair in pairs:
                if len(pair) != 3:
                    continue
                post = pair[0]
                response = pair[1]
                label = pair[2]
                if label == "good":
                    label = 1
                elif label == "bad":
                    label = 0
                if query and response:
                    dataset_list.append((query, post, response, label))
    print "length dataset:{}".format(len(dataset_list))
    return dataset_list

def cla_MAP(predicts):
    """计算map"""
    average_precision = 0
    # 遍历标准结果
    for i, query in enumerate(predicts.keys()):
        precision = 0
        count = 0
        pairs = predicts[query]
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        for index, pair in enumerate(pairs):
            # 检查是否匹配
            label = pair[-1]
            if label == 1:
                count += 1
                precision += float(count) / (index + 1)
        if len(predicts[query]) == 0 or count == 0:
            continue
        average_precision += float(precision)/count
    return float(average_precision)/len(predicts.keys())

def word2id(word_dict, dataset_list):
    """将数据转换成对应的id"""
    dataset_id = []
    if not dataset_list:
        return dataset_id
    def word_id_(words, word_dict):
        new_words = []
        for word in words:
            if word in word_dict:
                new_words.append(word_dict[word])
        return new_words
    for i, pair in enumerate(dataset_list):
        query = pair[0]
        post = pair[1]
        response = pair[2]
        label = pair[3]
        query_id = word_id_(query.strip().split(), word_dict)
        response_id = word_id_(response.strip().split(), word_dict)
        dataset_id.append((query_id, response_id))
    return dataset_id

def get_batch(dataset_id):
    for i, (d1, d2) in enumerate(dataset_id):
        X1 = np.zeros((1, config['data1_maxlen']), dtype=np.int32)
        X1_len = np.zeros((1,), dtype=np.int32)
        X2 = np.zeros((1, config['data2_maxlen']), dtype=np.int32)
        X2_len = np.zeros((1,), dtype=np.int32)
        X1[:] = config['words_num']
        X2[:] = config['words_num']
        d1_len = min(config['data1_maxlen'], len(d1))
        d2_len = min(config['data2_maxlen'], len(d2))
        X1[0, :d1_len], X1_len[0] = d1[:d1_len], d1_len
        X2[0, :d2_len], X2_len[0] = d2[:d2_len], d2_len
        yield X1, X1_len, X2, X2_len

def dynamic_pooling_index(len1, len2, max_len1, max_len2):
    def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
        stride1 = 1.0 * max_len1 / len1_one
        stride2 = 1.0 * max_len2 / len2_one
        idx1_one = [int(j/stride1) for j in range(max_len1)]
        idx2_one = [int(j/stride2) for j in range(max_len2)]
        mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
        index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2, 1, 0))
        return index_one
    index = []
    for i in range(len(len1)):
        index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))
    return np.array(index)

def save_predict(config, output_dict):
    """保存结果"""
    output_file = config["output_file"]
    ouput_dir, _ = os.path.split(output_file)
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    with codecs.open(output_file, "w", encoding="utf-8") as output_writer:
        for query, pairs in output_dict.items():
            temp_dict = {}
            temp_dict["query"] = query
            for pair in pairs:
                post = pair[0]
                response = pair[1]
                score = str(pair[2])
                label = str(pair[3])
                new_pairs = [post, response, score, "MP", label]
                if "pairs" not in temp_dict:
                    temp_dict["pairs"] = []
                temp_dict["pairs"].append(new_pairs)
            output_writer.write(json.dumps(temp_dict, ensure_ascii=False) + "\n")
    return None

def predict(config, dataset_name):
    word_dict_path = config["word_dict_path"]
    word_dict = load_word_dict(word_dict_path)
    words_len = len(word_dict)
    word_dict['[PAD]'] = words_len
    config['words_num'] = words_len

    # 获取batch数据
    test_data_path = config["test_data_path"]
    dataset_list = load_dataset(test_data_path, dataset_name)
    dataset_id = word2id(word_dict, dataset_list)
    test_generator = get_batch(dataset_id)

    # 加载模型
    model_tag = config["model_tag"]
    saved_model_dir = config["saved_model_dir"]
    with tf.Graph().as_default() as graph:
        sess = tf.Session()
        meta_graph_def = tf.saved_model.loader.load(sess, [model_tag], saved_model_dir)
        signature = meta_graph_def.signature_def
        query = graph.get_tensor_by_name(signature['model'].inputs['X1'].name)
        query_len = graph.get_tensor_by_name(signature['model'].inputs['X1_len'].name)
        doc = graph.get_tensor_by_name(signature['model'].inputs['X2'].name)
        doc_len = graph.get_tensor_by_name(signature['model'].inputs['X2_len'].name)
        dpool_index = graph.get_tensor_by_name(signature['model'].inputs['dpool_index'].name)
        predict = graph.get_tensor_by_name(signature['model'].outputs['predict'].name)
    # 预测
    query_maxlen = config["data1_maxlen"]
    doc_maxlen = config["data2_maxlen"]
    output_dict = {}
    count = 0
    for X1, X1_len, X2, X2_len in test_generator:
        dp_index = dynamic_pooling_index(X1_len, X2_len, query_maxlen, doc_maxlen)
        feed_dict = {query: X1, query_len: X1_len, doc: X2,
                     doc_len: X2_len, dpool_index: dp_index}
        pred = sess.run(predict, feed_dict=feed_dict)
        # pair(query, post, response, label)
        pair = dataset_list[count]
        query_raw = pair[0]
        post_raw = pair[1]
        response_raw = pair[2]
        label_raw = int(pair[3])
        score_raw = float(pred.tolist()[0][0])
        if query_raw not in output_dict:
            output_dict[query_raw] = []
        output_dict[query_raw].append((post_raw, response_raw, score_raw, label_raw))
        count += 1
    print "count: ", count
    MAP = cla_MAP(output_dict)
    save_predict(config, output_dict)
    print "MAP:", MAP
    print "done!"

if __name__ == "__main__":
    config = json.loads(open(sys.argv[1]).read())
    dataset_name = config["dataset_name"]
    predict(config, dataset_name)