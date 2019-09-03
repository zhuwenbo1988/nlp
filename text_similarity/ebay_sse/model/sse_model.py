# coding=utf-8

import tensorflow as tf

class SSE():
  def __init__(self, vector_size=1024, reshape_size=1024, learning_rate=0.0001, loss_weight=15.0):

    self.input_vector_size = vector_size
    self.M_size = reshape_size
    self.max_gradient_norm = 5.0
    self.learning_rate = learning_rate
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    self.src_input = tf.placeholder(tf.float32, [None, self.input_vector_size], name='src_vec')
    self.tgt_input = tf.placeholder(tf.float32, [None, self.input_vector_size], name='tgt_vec')
    self.labels = tf.placeholder(tf.float32, [None], name='labels')

    with tf.variable_scope('vector_mapper'):
      self.src_M = tf.get_variable('src_M', shape=[self.input_vector_size, self.M_size],
                                   initializer=tf.truncated_normal_initializer())
      self.tgt_M = tf.get_variable('tgt_M', shape=[self.input_vector_size, self.M_size],
                                   initializer=tf.truncated_normal_initializer())
      self.src_seq_embedding = tf.matmul(self.src_input, self.src_M, name='src_M_vec')
      self.tgt_seq_embedding = tf.matmul(self.tgt_input, self.tgt_M, name='tgt_M_vec')

    with tf.variable_scope('similarity'):
      self.norm_src_seq_embedding = tf.nn.l2_normalize(self.src_seq_embedding, dim=-1, name='src_M_norm_vec')
      self.norm_tgt_seq_embedding = tf.nn.l2_normalize(self.tgt_seq_embedding, dim=-1, name='tgt_M_norm_vec')

      self.similarity = tf.matmul(self.norm_src_seq_embedding, self.norm_tgt_seq_embedding, transpose_b=True, name='sim_1')

      self.binarylogit = tf.reduce_sum(tf.multiply(self.norm_src_seq_embedding, self.norm_tgt_seq_embedding), axis=-1, name='sim_2')

      self.predict_similarity = tf.sigmoid(64.0 * self.binarylogit, name='pred_sim')

    with tf.variable_scope('training_loss'):
      w = loss_weight
      self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.multiply(64.0, self.binarylogit), targets=self.labels, pos_weight=w), name='loss')

      self.train_acc = tf.reduce_mean(tf.multiply(self.labels, tf.floor(tf.sigmoid( 64.0 * self.binarylogit) + 0.1))) + tf.reduce_mean(tf.multiply(1.0 - self.labels, tf.floor(1.1 - tf.sigmoid( 64.0 * self.binarylogit))))

    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_gradient_norm )
    self.train = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=self.global_step)
