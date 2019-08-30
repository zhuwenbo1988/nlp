# coding=utf-8

import tensorflow as tf
tf.set_random_seed(1234)


class TextCNN(object):

    def __init__(self,
                 sequence_length,
                 sentiment_length,
                 patterns_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filter,
                 embeddings=None,
                 sentiment_embed_size=64,
                 pattern_embed_size=64,
                 l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(
            tf.int64, [None, sequence_length], name="input_x")
        self.labels = tf.placeholder(tf.int64, [None], name="labels")
        self.sentiments = tf.placeholder(
            tf.int64, [None, sentiment_length], name='sentiment')
        self.patterns = tf.placeholder(
            tf.int64, [None, patterns_length], name='patterns')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        self.sentiments_size = sentiment_embed_size
        self.pattern_embed_size = pattern_embed_size

        l2_loss = tf.constant(0.0, name="l2_loss")

        with tf.name_scope("embedding"):
            if embeddings is not None:
                self.embeddings = tf.Variable(embeddings, name="embeddings")
            else:
                self.embeddings = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="embeddings")

            self.embedding_x = tf.nn.embedding_lookup(
                self.embeddings, self.input_x)
            # fit the conv2d input size [batch,height,width,in_channel]
            self.embedding_x_expand = tf.expand_dims(self.embedding_x, -1)

        with tf.name_scope("sentiment"):
            self.sentiments_embeddings = tf.Variable(tf.random_uniform(
                [num_classes + 1, self.sentiments_size], -1.0, 1.0), name='sents_embeddings')
            sentiments = tf.nn.embedding_lookup(
                self.sentiments_embeddings, self.sentiments)
            sentiment = tf.reduce_max(sentiments, 1)

        with tf.name_scope("pattern"):
            self.patterns_embeddings = tf.Variable(
                tf.random_uniform([num_classes + 1, self.pattern_embed_size], -1.0, 1.0), name='patterns_embeddings')
            patterns_embed = tf.nn.embedding_lookup(
                self.patterns_embeddings, self.patterns)
            self.patt_regex = tf.reduce_max(patterns_embed, 1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # [filter_height,filter_width,in_channels,out_channels]
                filter_shape = [filter_size, embedding_size, 1, num_filter]
                W1 = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1, name="W1"))
                b1 = tf.Variable(tf.constant(
                    0.1, shape=[num_filter]), name="b1")

                # filter_shape = [filter_size, 1, num_filter, 2 * num_filter]
                # W2 = tf.Variable(tf.truncated_normal(
                #     filter_shape, stddev=0.1, name="W2"))
                # b2 = tf.Variable(tf.constant(
                #     0.1, shape=[2 * num_filter]), name="b2")

                conv1 = tf.nn.conv2d(
                    self.embedding_x_expand,
                    W1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv1"
                )
                # output_size : [batch,height,width,channels]
                # [?,sequence_length-filter_size+1,embedding_size,out_channels]
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
                pooled1 = tf.nn.max_pool(
                    h1,
                    #ksize=[1, 2, 1, 1],
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool1"
                )  # [batch,height,width,channels]

                # conv2 = tf.nn.conv2d(
                #     pooled1,
                #     W2,
                #     strides=[1, 1, 1, 1],
                #     padding="VALID",
                #     name="conv2"
                # )  # output_size : [batch,height,width,channels]
                # # [?,sequence_length-filter_size+1,embedding_size,out_channels]
                # h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
                # pooled2 = tf.nn.max_pool(
                #     h2,
                #     ksize=[1, sequence_length - filter_size * 2 + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool2"
                # )  # [batch,height,width,channels]

            pooled_outputs.append(pooled1)
        #num_filter_total = 2 * num_filter * len(filter_sizes)
        num_filter_total = num_filter * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])
        self.h_pool_flat_1 = self.h_pool_flat

        with tf.name_scope("concat"):
            self.h_pool_flat = tf.concat(
                [tf.concat([self.h_pool_flat, self.patt_regex], 1), sentiment], 1)

        with tf.name_scope("dropout"):
            self.h_pool = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_filter_total + pattern_embed_size + sentiment_embed_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.outputs = tf.nn.xw_plus_b(self.h_pool, W, b, name="outputs")

            self.logits = tf.nn.softmax(self.outputs, 1, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.outputs, labels=self.labels)  # outputs better than logits
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct = tf.equal(self.predictions, self.labels)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct, tf.float32), name="accuaracy")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(
            self.grads_and_vars, global_step=self.global_step)