# coding=utf-8

import tensorflow as tf
import configparser
import os
import sys

conf_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(conf_file)

model_dir = config.get('train', 'model_loc')

model_tag = 'sse'
saved_model_dir = os.path.join(model_dir, 'serving_model')
with tf.Graph().as_default() as graph:
    sess = tf.Session()
    meta_graph_def = tf.saved_model.loader.load(sess, [model_tag], saved_model_dir)

    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['src_vec', 'tgt_vec', 'similarity/pred_sim'])
    with tf.gfile.FastGFile('sse.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
