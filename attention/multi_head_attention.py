# coding=utf-8

import tensorflow as tf

def multi_head_attention(queries, keys, num_heads, attention_size, drop_rate=0.0, is_train=True, reuse=None,
                         scope=None):
    # borrowed from: https://github.com/Kyubyong/transformer/blob/master/modules.py
    with tf.variable_scope(scope or "multi_head_attention", reuse=reuse):
        if attention_size is None:
            attention_size = queries.get_shape().as_list()[-1]
        # linear projections, shape=(batch_size, max_time, attention_size)
        query = tf.layers.dense(queries, attention_size, activation=tf.nn.relu, name="query_project")
        key = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="key_project")
        value = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="value_project")
        # split and concatenation, shape=(batch_size * num_heads, max_time, attention_size / num_heads)
        query_ = tf.concat(tf.split(query, num_heads, axis=2), axis=0)
        key_ = tf.concat(tf.split(key, num_heads, axis=2), axis=0)
        value_ = tf.concat(tf.split(value, num_heads, axis=2), axis=0)
        # multiplication
        attn_outs = tf.matmul(query_, tf.transpose(key_, [0, 2, 1]))
        # scale
        attn_outs = attn_outs / (key_.get_shape().as_list()[-1] ** 0.5)
        # key masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # shape=(batch_size, max_time)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # shape=(batch_size * num_heads, max_time)
        # shape=(batch_size * num_heads, max_time, max_time)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(attn_outs) * (-2 ** 32 + 1)
        # shape=(batch_size, max_time, attention_size)
        attn_outs = tf.where(tf.equal(key_masks, 0), paddings, attn_outs)
        # activation
        attn_outs = tf.nn.softmax(attn_outs)
        # query masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        attn_outs *= query_masks
        # dropout
        attn_outs = tf.layers.dropout(attn_outs, rate=drop_rate, training=is_train)
        # weighted sum
        outputs = tf.matmul(attn_outs, value_)
        # restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        # 如果有问题,下面的两行代码可以不用
        outputs += queries  # residual connection
        outputs = layer_normalize(outputs)
    return outputs
