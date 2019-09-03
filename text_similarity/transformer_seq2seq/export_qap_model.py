# -*- coding: utf-8 -*-

import tensorflow as tf
from model import Transformer
from hparams import Hparams
from utils import load_hparams

ckpt_dir = 'log/1'

# 加载参数
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

load_hparams(hp, ckpt_dir)

with tf.Session() as sess:

    input_ids_p = tf.placeholder(tf.int32, [None, None], name="input_ids")
    input_len_p = tf.placeholder(tf.int32, [None], name="input_len")

    m = Transformer(hp)
    # tf.constant(1) is useless
    xs = (input_ids_p, input_len_p, tf.constant(1))
    memory, _, _ = m.encode(xs, False)
    
    vector = tf.reduce_mean(memory, axis=1, name='avg_vector')

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    graph_def = tf.get_default_graph().as_graph_def()
    # encoder/num_blocks_0/positionwise_feedforward/ln/add_1 is memory
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['input_ids', 'input_len', 'encoder/num_blocks_0/positionwise_feedforward/ln/add_1', 'avg_vector'])
    with tf.gfile.FastGFile('tsf.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())    
