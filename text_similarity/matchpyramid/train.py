# -*- coding:utf-8 -*-

import os
import sys
import json
import random
import codecs
import shutil
import numpy as np
import tensorflow as tf
from importlib import import_module
import data_utils as du
import pytextnet as pt

reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.insert(0, 'model/')

def eval_MAP(pred, gt):
    """计算map"""
    map_value = 0.0
    r = 0.0
    c = zip(pred, gt)
    random.shuffle(c)
    c = sorted(c, key=lambda x: x[0], reverse=True)
    for j, (p, g) in enumerate(c):
        if g != 0:
            r += 1
            map_value += r/(j+1.0)
    if r == 0:
        return 0.0
    else:
        return map_value/r

def load_embedding(config):
    basedir = config["data_dir"]
    word_dict_path = os.path.join(basedir, config["word_dict_path"])
    word_dict, iword_dict = pt.io.base.read_word_dict(word_dict_path)
    embedding_path = os.path.join(basedir, config["embedding_path"])
    embed_dict = pt.io.base.read_embedding(embedding_path)
    embed_size = config['embed_size']

    _PAD_ = len(word_dict)
    embed_dict[_PAD_] = np.zeros((embed_size,), dtype=np.float32)
    word_dict[_PAD_] = '[PAD]'
    iword_dict['[PAD]'] = _PAD_
    W_init_embed = np.float32(np.random.uniform(-0.02, 0.02, [len(word_dict), embed_size]))
    embedding = pt.io.base.convert_embed_2_numpy(embed_dict, embed=W_init_embed)
    return _PAD_, embedding

def train(config):
    basedir = config["data_dir"]
    log_file = config['log_file']
    flog = codecs.open(log_file, 'w', encoding='utf-8')

    _PAD_, embedding = load_embedding(config)
    config['words_num'] = _PAD_
    config['embedding'] = embedding

    mo = import_module(config['model_file'])
    model = mo.Model(config)
    sess = tf.Session()
    model.init_step(sess)

    # 查看loss
    tensorboard_dir = config["tensorboard"]
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    relation_train_file = os.path.join(basedir, config["relation_train_file"])
    relation_test_file = os.path.join(basedir, config["relation_test_file"])
    pair_gen = du.PairGenerator(rel_file=relation_train_file, config=config)
    list_gen = du.ListGenerator(rel_file=relation_test_file, config=config)

    # 加载corpus数据
    query_corpus_path = os.path.join(basedir, config["query_corpus_path"])
    doc_corpus_path = os.path.join(basedir, config["doc_corpus_path"])
    data_query = pt.io.base.read_data(filename=query_corpus_path)
    data_doc = pt.io.base.read_data(filename=doc_corpus_path)

    iters = config['train_iters']
    save_model_iters = int(config['save_model_iters'])
    display_interval = int(config['display_interval'])
    test_model_iters = int(config['test_model_iters'])
    for i in range(iters):
        X1, X1_len, X2, X2_len, Y = pair_gen.get_batch(data1=data_query, data2=data_doc)
        feed_dict = {model.X1: X1, model.X1_len: X1_len, model.X2: X2,
                     model.X2_len: X2_len, model.Y: Y}
        loss, merged = model.train_step(sess, feed_dict, merged_summary)
        if (i+1) % display_interval == 0:
            print >> flog, '[Train:%s]' % (i + 1), loss
            print '[Train:%s]' % (i + 1), loss
        writer.add_summary(merged, (i + 1))
        flog.flush()
        # 保存模型
        if (i+1) % save_model_iters == 0:
            saved_model_dir = config["saved_model_dir"]
            saved_model_dir = saved_model_dir + ".{}".format(i+1)
            if os.path.exists(saved_model_dir):
                shutil.rmtree(saved_model_dir)
            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
            signature = tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'X1': model.X1, "X1_len": model.X1_len, "X2": model.X2,
                        "X2_len": model.X2_len, "dpool_index": model.dpool_index},
                outputs={'predict': model.pred})
            model_tag = config["model_tag"]
            builder.add_meta_graph_and_variables(sess,
                                                 [model_tag],
                                                 signature_def_map={
                                                     "model": signature
                                                 })
            builder.save()
        # 验证
        if (i+1) % test_model_iters == 0:
            map_v = 0.0
            map_c = 0.0
            test_generator = list_gen.get_batch(data_query, data_doc)
            for X1, X1_len, X2, X2_len, Y, pairs in test_generator:
                feed_dict = {model.X1: X1, model.X1_len: X1_len, model.X2: X2,
                             model.X2_len: X2_len, model.Y: Y}
                pred = model.test_step(sess, feed_dict)
                map_o = eval_MAP(pred, Y)
                map_v += map_o
                map_c += 1.0
            map_v /= map_c
            print >> flog, '[Test:%s]' % (i + 1), map_v
            print '[Test:%s]' % (i + 1), map_v
            flog.flush()
    flog.close()

if __name__ == "__main__":
    config = json.loads(open(sys.argv[1]).read())
    train(config)

