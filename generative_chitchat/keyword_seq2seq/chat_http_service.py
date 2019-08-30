#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import jieba
import jieba.analyse
import re
import sys
import os
import argparse
import json
from datetime import datetime

from models import model_factory
from models import model_helper
from models import vocab_utils as vocab
from config import Config
from http_utils import http_server_py3


def load_stop_words(model_path):
    stop_words = {}
    for line in open(os.path.join(model_path, 'stop_words')):
        stop_words[line.strip()] = 1
    return stop_words


def load_keyword_label(model_path):
    word2label = {}
    for line in open(os.path.join(model_path, 'word2label')):
        line = line.strip()
        items = line.split('\t')
        w = items[0]
        label = int(items[1])
        word2label[w] = label
    return word2label


def load_top_words(model_path):
    label2topn = {}
    for line in open(os.path.join(model_path, 'top_keywords')):
        line = line.strip()
        items = line.split('\t')
        label = int(items[0])
        words = items[1]
        label2topn[label] = words
    return label2topn


print('init config params')
args = vars()
args['mode'] = 'infer'
args['model_dir'] = 'service_model'
args['beam_width'] = 5
args['length_penalty_weight'] = 0.8
config = Config(**args)

stop_words = load_stop_words(config.model_dir)
word2label = load_keyword_label(config.model_dir)
label2topn = load_top_words(config.model_dir)

model = model_factory.create_model(config)

infer_model = model.create_infer_model_graph()
config_proto = model_helper.get_config_proto(config.log_device)

sess = tf.InteractiveSession(graph=infer_model.graph, config=config_proto)

ckpt = tf.train.latest_checkpoint(config.model_dir)
loaded_infer_model = model_helper.load_model(infer_model.model, ckpt, sess, "infer")


def extract_keyword(query):
    tags = jieba.analyse.extract_tags(query, topK=10)
    keyword = []
    for w in tags:
        if re.search('^\d+$', w):
                continue
        if re.search('^[a-zA-Z]+$', w):
                continue
        if w in stop_words:
                continue
        keyword.append(w)
    if not keyword:
        return None, None
    label = None
    word = None
    for w in keyword:
            if w in word2label:
                    label = word2label[w]
                    word = w
                    break
    if label:
        return word, label2topn[label]
    return None, None


def error(err_msg):
    rlt_json = {}
    rlt_json["status"] = "error"
    rlt_json['msg'] = err_msg
    return rlt_json


def success(response, cost):
    rlt_json = {}
    rlt_json["status"] = "success"
    rlt_json["response"] = response
    rlt_json["time_cost"] = cost
    return rlt_json


def chat(params):
    print(params)
    query = None
    if not 'query' in params:
        return json.dumps(error("need query"), ensure_ascii=False)
    query = params['query']
    if not query:
        return json.dumps(error("query is illegal"), ensure_ascii=False)

    beging_t = datetime.now()
    
    words = ' '.join(jieba.cut(query))
    keyword, n_keywords = extract_keyword(words)
    print('keyword :', keyword)
    print('extend keywords :', n_keywords)
    
    if not n_keywords:
        cost = (datetime.now() - beging_t).total_seconds()*1000
        return json.dumps(success('null', cost), ensure_ascii=False)
        
    inp = '{}  |  {}'.format(words, ' '.join(n_keywords))         
    iterator_feed_dict = {
        infer_model.src_placeholder: [inp],
        infer_model.batch_size_placeholder: 1,
    }
    sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
    output, _ = loaded_infer_model.decode(sess)

    if config.beam_width > 0:
        # get the top translation.
        output = output[0]

    resp = vocab.get_translation(output, sent_id=0)
    resp = resp.decode('utf-8')
    
    print('response :', resp)
    if '哈哈哈' in resp:
        cost = (datetime.now() - beging_t).total_seconds()*1000
        return json.dumps(success('null', cost), ensure_ascii=False)
    if '<unk>' in resp:
        resp = resp.replace('<unk>', keyword)
    cost = (datetime.now() - beging_t).total_seconds()*1000
    rlt_json = success(resp, cost)
    rlt_json['keyword'] = keyword
    rlt_json['extend_keywords'] = ' '.join(n_keywords)
    return json.dumps(rlt_json, ensure_ascii=False)


def Statuscheck(params):
    rlt_json = {}
    rlt_json["status"] = "ok"
    return json.dumps(rlt_json, ensure_ascii=False)


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except:
        print("USAGE: chatbot_service port")
        sys.exit(-1)
    HOST, PORT = "", port
    server = http_server_py3.HttpServer(HOST, PORT)
    server.Register("/chat", chat)
    server.Register("/status", Statuscheck)
    # TODO
    server.Register("/chat/", chat)
    server.Register("/status/", Statuscheck)
    print('running')
    try:
        server.Start()
    except Exception as e:
        pass
    finally:
        print('close tf session')
        sess.close()
