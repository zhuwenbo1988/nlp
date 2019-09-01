# coding=utf-8

import tensorflow as tf
import os
import sys
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

saved_model_dir = sys.argv[1]
model_type = sys.argv[2]

if model_type == 'ecm':
    with tf.Graph().as_default() as graph:
        sess = tf.Session()
        meta_graph_def = tf.saved_model.loader.load(sess, tf.saved_model.tag_constants.SERVING, saved_model_dir)
        graph_def = tf.get_default_graph().as_graph_def()
        for name in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            print(name)
    
        node_names = []
        node_names.append('seq2seq_placeholder/encoder_inputs')
        node_names.append('seq2seq_placeholder/encoder_length')
        node_names.append('seq2seq_placeholder/emotion_category')
        node_names.append('seq2seq_decoder/transpose')
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, node_names)
        with tf.gfile.FastGFile('ecm.pb', mode='wb') as f:
            f.write(output_graph_def.SerializeToString())

if model_type == 'seq2seq':
    with tf.Graph().as_default() as graph:
        sess = tf.Session()
        meta_graph_def = tf.saved_model.loader.load(sess, tf.saved_model.tag_constants.SERVING, saved_model_dir)
        graph_def = tf.get_default_graph().as_graph_def()
        for name in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            print(name)

        node_names = []
        node_names.append('seq2seq_placeholder/encoder_inputs')
        node_names.append('seq2seq_placeholder/encoder_length')
        node_names.append('seq2seq_decoder/transpose')
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, node_names)
        with tf.gfile.FastGFile('seq2seq_attn.pb', mode='wb') as f:
            f.write(output_graph_def.SerializeToString())
