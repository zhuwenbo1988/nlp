#! usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import json
import logging
import numpy as np

import tensorflow as tf
import codecs
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import estimator
import modeling
import optimization
import tokenization
from lstm_crf_layer import BLSTM_CRF

import tf_metrics
import pickle

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", '',
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", 'bert_config.json',
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", '', "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", 'output',
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", 'bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_bool(
    "save_tf_serving_model", False,
    "Whether to save tf serving model"
)

flags.DEFINE_bool(
    "crf_only", False,
    "Whether to use crf only."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_boolean('clean', True, 'remove the files which created by last training')

flags.DEFINE_bool("do_train", True, "Whether to run training."
                  )
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_export", True, "export tensorserving model")

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 15.0, "Total number of training epochs to perform.")
flags.DEFINE_float('droupout_rate', 0.9, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", 'vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_string('data_config_path', 'data.conf',
                    'data config file, which save train and dev config')
# lstm parame
flags.DEFINE_integer('lstm_size', 512, 'size of lstm units')
flags.DEFINE_integer('num_layers', 1, 'number of rnn layers, default is 1')
flags.DEFINE_string('cell', 'lstm', 'which rnn cell used')

# 词表特征
flags.DEFINE_integer('max_feature_length', 5, '')
flags.DEFINE_integer('feature_embedding_size', 300, '')
flags.DEFINE_bool("use_domain_intent_lexicon", True, "domain&intent cls features")
flags.DEFINE_bool("use_slot_lexicon", True, "slot tagging features")
flags.DEFINE_bool("use_domain_vector", False, "[CLS] token's embedding")


class InputExample(object):
    """A single training/test example."""

    def __init__(self, guid, text, domain_intent_label=None, domain_intent_lexicon_feature=None, slot_label=None, slot_lexicon_feature=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """
        self.guid = guid
        self.text = text
        self.domain_intent_label = domain_intent_label
        self.domain_intent_lexicon_feature = domain_intent_lexicon_feature
        self.slot_label = slot_label
        self.slot_lexicon_feature = slot_lexicon_feature

        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, domain_intent_label_id, domain_intent_lexicon_ids, slot_label_ids, slot_lexicon_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.domain_intent_label_id = domain_intent_label_id
        self.domain_intent_lexicon_ids = domain_intent_lexicon_ids
        self.slot_label_ids = slot_label_ids
        self.slot_lexicon_ids = slot_lexicon_ids
    
    
class DataProcessor(object):
    """Base class for data converters for sequence data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                o = json.loads(line.strip())
                lines.append(o)
        return lines

class E2eLuProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train_data.json")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev_data.json")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test_data.json")), "test")
    

    def get_labels(self):
        domain_intent_labels = []
        slot_labels = []
        domain_intent_lexicon_names = []
        slot_lexicon_names = []
        
        for line in open(os.path.join(FLAGS.data_dir, "domain_intent_labels.txt"), 'r'):
            domain_intent_labels.append(line.strip())
            
        for line in open(os.path.join(FLAGS.data_dir, "slot_labels.txt"), 'r'):
            slot_labels.append(line.strip())
            
        for line in open(os.path.join(FLAGS.data_dir, "domain_intent_lexicon_names.txt"), 'r'):
            domain_intent_lexicon_names.append(line.strip())
            
        for line in open(os.path.join(FLAGS.data_dir, "slot_lexicon_names.txt"), 'r'):
            slot_lexicon_names.append(line.strip()) 
        
        return domain_intent_labels, slot_labels, domain_intent_lexicon_names, slot_lexicon_names

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            json_obj = line
            
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(json_obj['text'])
            domain_intent_label = tokenization.convert_to_unicode(json_obj['domain_intent_label'])
            domain_intent_lexicon_feature = tokenization.convert_to_unicode(json_obj['domain_intent_lexicon_feature'])
            slot_label = tokenization.convert_to_unicode(json_obj['slot_label'])
            slot_lexicon_feature = tokenization.convert_to_unicode(json_obj['slot_lexicon_feature'])
            
            examples.append(InputExample(guid=guid, text=text, domain_intent_label=domain_intent_label, domain_intent_lexicon_feature=domain_intent_lexicon_feature, slot_label=slot_label, slot_lexicon_feature=slot_lexicon_feature))
        return examples
    
    
def convert_single_example(ex_index, 
                           example, 
                           domain_intent_label_list, 
                           slot_label_list, 
                           domain_intent_lexicon_name_list,
                           slot_lexicon_name_list, 
                           max_seq_length, 
                           max_feature_length, 
                           tokenizer, 
                           mode):
    # domain_intent vocab
    domain_intent_label_map = {}
    # index从0开始
    for (i, label) in enumerate(domain_intent_label_list):
        domain_intent_label_map[label] = i
        
    # slot vocab
    slot_label_map = {}
    # index从1开始, 0是padding位
    for (i, label) in enumerate(slot_label_list, 1):
        slot_label_map[label] = i
        
    # domain&intent cls lexicon
    domain_intent_lexicon_name_map = {}
    # index从1开始, 0是padding位
    for (i, label) in enumerate(domain_intent_lexicon_name_list, 1):
        domain_intent_lexicon_name_map[label] = i
    domain_intent_lexicon_name_map['[PAD]'] = 0
    
    # slot tagging lexicon
    slot_lexicon_name_map = {}
    # index从1开始, 0是padding位
    for (i, label) in enumerate(slot_lexicon_name_list, 1):
        slot_lexicon_name_map[label] = i
    slot_lexicon_name_map['[PAD]'] = 0
        
    # token -> wordpiece
    raw_tokens = list(example.text)
    raw_slot_labels = example.slot_label.split(' ')
    tokens = []
    slot_labels = []
    for i, word in enumerate(raw_tokens):
        # 如果是中文，word是字
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        slot_labels.append(raw_slot_labels[i])
            
    # 每个token都有slot词表特征
    raw_slot_features = example.slot_lexicon_feature.split('\t')
    slot_features = []
    for i in range(len(raw_tokens)):
        raw_feature = raw_slot_features[i]
        if raw_feature == 'O_VD':
            slot_features.append(['[PAD]'] * max_feature_length)
            continue
        tmp = []
        for name in slot_lexicon_name_list:
            if name in raw_feature:
                tmp.append(name)
        if len(tmp) >= max_feature_length:
            tmp = tmp[:max_feature_length]
        else:
            tmp.extend(['[PAD]'] * (max_feature_length - len(tmp)))
        slot_features.append(tmp)
        
    # 截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        # -2是合理的
        slot_labels = slot_labels[0:(max_seq_length - 2)]
        slot_features = slot_features[0:(max_seq_length - 2)]
    
    segment_ids = []
    slot_label_ids = []
    segment_ids.append(0)
    slot_lexicon_ids = []
    for i, token in enumerate(tokens):
        segment_ids.append(0)
        slot_label_ids.append(slot_label_map[slot_labels[i]])
        slot_feature = slot_features[i]
        tmp = []
        for name in slot_feature:
            tmp.append(slot_lexicon_name_map[name])
        slot_lexicon_ids.append(tmp)
        
    segment_ids.append(0)
    # slot加结尾
    slot_label_ids.append(slot_label_map["[SEP]"])
    
    tokens.insert(0, "[CLS]")  # 句子开始设置CLS 标志    
    tokens.append("[SEP]")  # 句尾添加[SEP] 标志
    
    # token -> id
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens.
    input_mask = [1] * len(input_ids)    

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        tokens.append("[PAD]")
    
    while len(slot_label_ids) < max_seq_length-1:
        slot_label_ids.append(0)
        
    while len(slot_lexicon_ids) < max_seq_length-1:
        slot_lexicon_ids.append([0] * max_feature_length)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(slot_label_ids) == max_seq_length-1
    assert len(slot_lexicon_ids) == max_seq_length-1
    assert len(slot_lexicon_ids[0]) == max_feature_length
            
    domain_intent_label = example.domain_intent_label
    domain_intent_label_id = domain_intent_label_map[domain_intent_label]
    
    # domain&intent词表特征是sentence级别的
    domain_intent_lexicon_ids = []
    feature_names = example.domain_intent_lexicon_feature.split(' ')
    for name in feature_names:
        if name in domain_intent_lexicon_name_map:
            domain_intent_lexicon_ids.append(domain_intent_lexicon_name_map[name])
    if len(domain_intent_lexicon_ids) >= max_feature_length:
        domain_intent_lexicon_ids = domain_intent_lexicon_ids[:max_feature_length]
    else:
        domain_intent_lexicon_ids.extend([domain_intent_lexicon_name_map['[PAD]']] * (max_feature_length - len(domain_intent_lexicon_ids)))
    
    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % ",".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % ",".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % ",".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % ",".join([str(x) for x in segment_ids]))
        tf.logging.info("domain_intent label: %s (id = %d)" % (example.domain_intent_label, domain_intent_label_id))
        tf.logging.info("domain_intent_lexicon_ids: %s" % ",".join([str(x) for x in domain_intent_lexicon_ids]))
        tf.logging.info("slot_label_ids: %s" % ",".join([str(x) for x in slot_label_ids]))
        for i in range(len(slot_lexicon_ids)):
            tf.logging.info("slot_lexicon_ids: %s" % ",".join([str(x) for x in slot_lexicon_ids[i]]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        domain_intent_label_id=domain_intent_label_id,
        domain_intent_lexicon_ids=domain_intent_lexicon_ids,
        slot_label_ids=slot_label_ids,
        slot_lexicon_ids=slot_lexicon_ids
    )
    
    return feature    
    

def filed_based_convert_examples_to_features(examples, 
                                             domain_intent_label_list, 
                                             slot_label_list, 
                                             domain_intent_lexicon_name_list,
                                             slot_lexicon_name_list, 
                                             max_seq_length, 
                                             max_feature_length,
                                             tokenizer, 
                                             output_file, 
                                             mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 处理每条训练数据
        feature = convert_single_example(ex_index, 
                                         example, 
                                         domain_intent_label_list, 
                                         slot_label_list, 
                                         domain_intent_lexicon_name_list,
                                         slot_lexicon_name_list,
                                         max_seq_length, 
                                         max_feature_length,
                                         tokenizer, 
                                         mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        # list
        features["domain_intent_label_id"] = create_int_feature([feature.domain_intent_label_id])
        features["domain_intent_lexicon_ids"] = create_int_feature(feature.domain_intent_lexicon_ids)
        features["slot_label_ids"] = create_int_feature(feature.slot_label_ids)
        # 展开
        features["slot_lexicon_ids"] = create_int_feature(np.array(feature.slot_lexicon_ids).ravel())
        
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    
    
def file_based_input_fn_builder(input_file, seq_length, feature_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "domain_intent_label_id": tf.FixedLenFeature([], tf.int64),
        "domain_intent_lexicon_ids": tf.FixedLenFeature([feature_length], tf.int64),
        # -1是因为没有[CLS] token
        "slot_label_ids": tf.FixedLenFeature([seq_length-1], tf.int64),
        "slot_lexicon_ids": tf.FixedLenFeature([(seq_length-1)*feature_length], tf.int64)
    }
    
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            # 恢复
            if name == 'slot_lexicon_ids':
                t = tf.reshape(t, (seq_length-1, feature_length))   
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def multi_head_attention(queries, keys, num_heads=1, attention_size=None, drop_rate=0.0, is_train=True, reuse=None,
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
    return outputs


def create_model(bert_config, 
                 is_training,
                 input_ids, 
                 input_mask, 
                 segment_ids,
                 use_one_hot_embeddings,
                 domain_intent_lexicon_ids,
                 slot_lexicon_ids,
                 input_domain_intent_label,
                 num_domain_intent_labels,
                 input_slot_labels,
                 num_slot_labels,
                 num_domain_intent_lexicon,
                 num_slot_lexicon):
    """
    创建X模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    
    tokens_embedding = model.get_sequence_output()[:,1:,:]
    max_seq_length = tokens_embedding.shape[1].value
    # 等于0 -> 0 ; 大于0 -> 1
    used = tf.sign(tf.abs(input_ids))[:, 1:]

    first_token_embedding = None
    if FLAGS.use_domain_vector:
        first_token_embedding = model.get_sequence_output()[:,0,:]
        first_token_embedding = tf.expand_dims(first_token_embedding, axis=1)
        first_token_embedding = tf.tile(first_token_embedding, [1, max_seq_length, 1])
    
    # [batch_size] 大小的向量，包含了当前batch中的序列长度
    lengths = tf.reduce_sum(used, reduction_indices=1)
    
    # 判断batch size
    batch_size = tokens_embedding.shape[0].value
    if not batch_size:
        # test
        batch_size = 1
    print('batch size is {}'.format(batch_size))
    
    if FLAGS.use_slot_lexicon: 
        # slot 词表特征
        slot_feature_embedding_table = tf.Variable(tf.random_uniform([num_slot_lexicon, FLAGS.feature_embedding_size], -1.0, 1.0), name="slot_feature_embeddings")
        slot_lexicon_embedding = tf.nn.embedding_lookup(slot_feature_embedding_table, slot_lexicon_ids)
        # aggregrate slot features
        slot_lexicon_embedding_pooled = tf.reshape(slot_lexicon_embedding, [batch_size*(FLAGS.max_seq_length-1), -1])
        W_slot = tf.get_variable("slot_lexicon_output_weights", [FLAGS.feature_embedding_size*FLAGS.max_feature_length, 768], initializer=tf.truncated_normal_initializer(stddev=0.02))
        slot_lexicon_embedding_pooled = tf.matmul(slot_lexicon_embedding_pooled, W_slot, transpose_b=False)
        slot_lexicon_embedding_pooled = tf.reshape(slot_lexicon_embedding_pooled, [batch_size, FLAGS.max_seq_length-1, -1])
        # self attention
        slot_lexicon_embedding_pooled = multi_head_attention(slot_lexicon_embedding_pooled, slot_lexicon_embedding_pooled)
        # concat bert embedding and slot features
        tokens_embedding = tf.concat([tokens_embedding, slot_lexicon_embedding_pooled], 2)
    
    blstm_crf = BLSTM_CRF(embedded_chars=tokens_embedding, 
                          hidden_unit=FLAGS.lstm_size, 
                          cell_type=FLAGS.cell, 
                          num_layers=FLAGS.num_layers, 
                          dropout_rate=FLAGS.droupout_rate, 
                          initializers=initializers, 
                          num_labels=num_slot_labels, 
                          seq_length=max_seq_length, 
                          labels=input_slot_labels, 
                          lengths=lengths, 
                          is_training=is_training,
                          domain_vector=first_token_embedding)
    slot_loss, slot_logits, slot_trans, slot_pred_ids = blstm_crf.add_blstm_crf_layer(crf_only=FLAGS.crf_only)
    
    # 获取整个句子的embedding
    sentence_embedding = model.get_pooled_output()
    hidden_size = sentence_embedding.shape[-1].value
    if FLAGS.use_domain_intent_lexicon:
        # 让domain&intent features与bert embedding有同样的权重
        hidden_size += hidden_size
    
    # domain_intent分类
    # W
    domain_intent_output_weights = tf.get_variable(
      "domain_intent_output_weights", [num_domain_intent_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
    # b
    domain_intent_output_bias = tf.get_variable(
      "domain_intent_output_bias", [num_domain_intent_labels], initializer=tf.zeros_initializer())
    with tf.variable_scope("domain_intent_cls_loss"):
        if is_training:
            # I.e., 0.1 dropout
            sentence_embedding = tf.nn.dropout(sentence_embedding, keep_prob=0.9)
        
        if FLAGS.use_domain_intent_lexicon:
            domain_intent_feature_embedding_table = tf.Variable(tf.random_uniform([num_domain_intent_lexicon, FLAGS.feature_embedding_size], -1.0, 1.0), name="domain_intent_feature_embeddings")

            domain_intent_lexicon_embedding = tf.nn.embedding_lookup(domain_intent_feature_embedding_table, domain_intent_lexicon_ids)
            # aggregrate domain&intent features
            domain_intent_lexicon_embedding_pooled = tf.reshape(domain_intent_lexicon_embedding, [batch_size, -1])
            W = tf.get_variable("domain_intent_lexicon_output_weights", [FLAGS.feature_embedding_size*FLAGS.max_feature_length, hidden_size/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
            domain_intent_lexicon_embedding_pooled = tf.matmul(domain_intent_lexicon_embedding_pooled, W, transpose_b=False)
            # concat bert embedding and domain&intent features
            sentence_embedding = tf.concat([sentence_embedding, domain_intent_lexicon_embedding_pooled], 1)
            
        domain_intent_logits = tf.matmul(sentence_embedding, domain_intent_output_weights, transpose_b=True)
        domain_intent_logits = tf.nn.bias_add(domain_intent_logits, domain_intent_output_bias)
        domain_intent_probabilities = tf.nn.softmax(domain_intent_logits, axis=-1, name="domain_intent_probabilities")
        domain_intent_log_probs = tf.nn.log_softmax(domain_intent_logits, axis=-1)
        one_hot_labels = tf.one_hot(input_domain_intent_label, depth=num_domain_intent_labels, dtype=tf.float32)
        per_example_domain_intent_loss = -tf.reduce_sum(one_hot_labels * domain_intent_log_probs, axis=-1)
        domain_intent_loss = tf.reduce_mean(per_example_domain_intent_loss)
    
    return domain_intent_loss, per_example_domain_intent_loss, domain_intent_logits, domain_intent_probabilities, slot_loss, slot_logits, slot_trans, slot_pred_ids


def domain_intent_model_fn_builder(bert_config, 
                            num_domain_intent_labels, 
                            num_slot_labels, 
                            num_domain_intent_lexicon,
                            num_slot_lexicon,
                            init_checkpoint, 
                            learning_rate, 
                            num_train_steps, 
                            num_warmup_steps, 
                            use_tpu, 
                            use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        domain_intent_label_id = features["domain_intent_label_id"]
        domain_intent_lexicon_ids = features["domain_intent_lexicon_ids"]
        slot_label_ids = features["slot_label_ids"]
        slot_lexicon_ids = features["slot_lexicon_ids"]
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        domain_intent_loss, per_example_domain_intent_loss, domain_intent_logits, domain_intent_prob, \
        slot_loss, slot_logits, slot_trans, slot_pred_ids = create_model(bert_config, 
                                                            is_training, 
                                                            input_ids, 
                                                            input_mask, 
                                                            segment_ids, 
                                                            use_one_hot_embeddings, 
                                                            domain_intent_lexicon_ids,
                                                            slot_lexicon_ids, 
                                                            domain_intent_label_id, 
                                                            num_domain_intent_labels, 
                                                            slot_label_ids, 
                                                            num_slot_labels,
                                                            num_domain_intent_lexicon,
                                                            num_slot_lexicon)
        with tf.variable_scope("joint_domain_intent_slot_loss"):
            joint_domain_intent_slot_loss = domain_intent_loss + slot_loss
            
        with tf.variable_scope("domain_intent_cls_op"):
            domain_intent_train_op = optimization.create_optimizer(domain_intent_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
        with tf.variable_scope("slot_cls_op"):
            slot_train_op = optimization.create_optimizer(slot_loss, learning_rate, 0, 0, use_tpu)
        with tf.variable_scope("domain_intent_slot_cls_op"):
            domain_intent_slot_train_op = optimization.create_optimizer(joint_domain_intent_slot_loss, learning_rate, 0, 0, use_tpu)
        
        tvars = tf.trainable_variables()
        scaffold_fn = None
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                # gpu
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            logging_hook = tf.train.LoggingTensorHook({"domain_intent_loss" : domain_intent_loss, 
                }, every_n_iter=10)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=domain_intent_loss,
              train_op=domain_intent_train_op,
              scaffold_fn=scaffold_fn,
              training_hooks = [logging_hook])
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=domain_intent_prob,
                scaffold_fn=scaffold_fn
            )
        return output_spec
        
    return model_fn


def slot_model_fn_builder(bert_config, 
                          num_domain_intent_labels, 
                          num_slot_labels, 
                          num_domain_intent_lexicon,
                          num_slot_lexicon,
                          init_checkpoint, 
                          learning_rate, 
                          num_train_steps, 
                          num_warmup_steps, 
                          use_tpu, 
                          use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        domain_intent_label_id = features["domain_intent_label_id"]
        domain_intent_lexicon_ids = features["domain_intent_lexicon_ids"]
        slot_label_ids = features["slot_label_ids"]
        slot_lexicon_ids = features["slot_lexicon_ids"]
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        domain_intent_loss, per_example_domain_intent_loss, domain_intent_logits, domain_intent_prob, \
        slot_loss, slot_logits, slot_trans, slot_pred_ids = create_model(bert_config, 
                                                                         is_training, 
                                                                         input_ids, 
                                                                         input_mask, 
                                                                         segment_ids, 
                                                                         use_one_hot_embeddings, 
                                                                         domain_intent_lexicon_ids,
                                                                         slot_lexicon_ids, 
                                                                         domain_intent_label_id, 
                                                                         num_domain_intent_labels, 
                                                                         slot_label_ids, 
                                                                         num_slot_labels,
                                                                         num_domain_intent_lexicon,
                                                                         num_slot_lexicon,)
        with tf.variable_scope("joint_domain_intent_slot_loss"):
            joint_domain_intent_slot_loss = domain_intent_loss + slot_loss
            
        with tf.variable_scope("domain_intent_cls_op"):
            domain_intent_train_op = optimization.create_optimizer(domain_intent_loss, learning_rate, 0, 0, use_tpu)
        with tf.variable_scope("slot_cls_op"):
            slot_train_op = optimization.create_optimizer(slot_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
        with tf.variable_scope("domain_intent_slot_cls_op"):
            domain_intent_slot_train_op = optimization.create_optimizer(joint_domain_intent_slot_loss, learning_rate, 0, 0, use_tpu)
            
        tvars = tf.trainable_variables()
        scaffold_fn = None
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            logging_hook = tf.train.LoggingTensorHook({"slot_loss" : slot_loss}, every_n_iter=10)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=slot_loss,
                train_op=slot_train_op,
                scaffold_fn=scaffold_fn,
                training_hooks = [logging_hook])
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=slot_pred_ids,
                scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def domain_and_intent_and_slot_model_fn_builder(bert_config, 
                                                num_domain_intent_labels, 
                                                num_slot_labels, 
                                                num_domain_intent_lexicon,
                                                num_slot_lexicon,
                                                init_checkpoint, 
                                                learning_rate, 
                                                num_train_steps, 
                                                num_warmup_steps, 
                                                use_tpu, 
                                                use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        domain_intent_label_id = features["domain_intent_label_id"]
        domain_intent_lexicon_ids = features["domain_intent_lexicon_ids"]
        slot_label_ids = features["slot_label_ids"]
        slot_lexicon_ids = features["slot_lexicon_ids"]
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        domain_intent_loss, per_example_domain_intent_loss, domain_intent_logits, domain_intent_prob, \
        slot_loss, slot_logits, slot_trans, slot_pred_ids = create_model(bert_config, 
                                                                         is_training, 
                                                                         input_ids, 
                                                                         input_mask, 
                                                                         segment_ids, 
                                                                         use_one_hot_embeddings, 
                                                                         domain_intent_lexicon_ids,
                                                                         slot_lexicon_ids, 
                                                                         domain_intent_label_id, 
                                                                         num_domain_intent_labels, 
                                                                         slot_label_ids, 
                                                                         num_slot_labels,
                                                                         num_domain_intent_lexicon,
                                                                         num_slot_lexicon)
        with tf.variable_scope("joint_domain_intent_slot_loss"):
            joint_domain_intent_slot_loss = domain_intent_loss + slot_loss
            
        with tf.variable_scope("domain_intent_cls_op"):
            domain_intent_train_op = optimization.create_optimizer(domain_intent_loss, learning_rate, 0, 0, use_tpu)
        with tf.variable_scope("slot_cls_op"):
            slot_train_op = optimization.create_optimizer(slot_loss, learning_rate, 0, 0, use_tpu)
        with tf.variable_scope("domain_intent_slot_cls_op"):
            domain_intent_slot_train_op = optimization.create_optimizer(joint_domain_intent_slot_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
        
        tvars = tf.trainable_variables()
        scaffold_fn = None
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                # gpu
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            logging_hook = tf.train.LoggingTensorHook({"domain_intent_loss" : domain_intent_loss,
                                                       "slot_loss" : slot_loss, 
                                                       "joint_domain_intent_slot_loss" : joint_domain_intent_slot_loss
                }, every_n_iter=10)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=joint_domain_intent_slot_loss,
              train_op=domain_intent_slot_train_op,
              scaffold_fn=scaffold_fn,
              training_hooks = [logging_hook])
            
            
            domain_intent_loss, per_example_domain_intent_loss, domain_intent_logits, domain_intent_prob, \
            
            
        if mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_domain_intent_loss, domain_intent_label_id, domain_intent_logits, 
                         slot_label_ids, slot_pred_ids):
                # domain
                domain_intent_preds = tf.argmax(domain_intent_logits, axis=-1, output_type=tf.int32)
                domain_intent_eval_acc = tf.metrics.accuracy(domain_intent_label_id, domain_intent_preds)
                domain_intent_eval_loss = tf.metrics.mean(per_example_domain_intent_loss)
                # slot
                # 首先对结果进行维特比解码
                # crf 解码
                weight = tf.sequence_mask(FLAGS.max_seq_length)
                slot_eval_prec = tf_metrics.precision(slot_label_ids, slot_pred_ids, num_slot_labels)
                slot_eval_rec = tf_metrics.recall(slot_label_ids, slot_pred_ids, num_slot_labels)
                slot_eval_f = tf_metrics.f1(slot_label_ids, slot_pred_ids, num_slot_labels)
                slot_eval_loss = tf.metrics.mean_squared_error(labels=slot_label_ids, predictions=slot_pred_ids)
                return {
                    "domain_intent_eval_accuracy": domain_intent_eval_acc,
                    "domain_intent_eval_loss": domain_intent_eval_loss,
                    "slot_eval_precision": slot_eval_prec,
                    "slot_eval_recall": slot_eval_rec,
                    "slot_eval_f1": slot_eval_f,
                    "slot_eval_loss": slot_eval_loss
                }
            eval_metrics = (metric_fn, [per_example_domain_intent_loss, domain_intent_label_id, domain_intent_logits, 
                                       slot_label_ids, slot_pred_ids])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=joint_domain_intent_slot_loss,
              eval_metrics=eval_metrics,
              scaffold_fn=scaffold_fn)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            predict_result = {}
            predict_result['domain_intent_result'] = domain_intent_prob
            predict_result['slot_result'] = slot_pred_ids
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predict_result,
                scaffold_fn=scaffold_fn
            )
            if FLAGS.do_export:
                gpu_config = tf.ConfigProto()
                gpu_config.gpu_options.allow_growth = True
                sess=tf.Session(config=gpu_config)
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.output_dir))
        
                export_dir = './serving_model'
        
                builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
                signature = tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs={"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "domain_intent_lexicon_ids": domain_intent_lexicon_ids, "slot_lexicon_ids": slot_lexicon_ids}, outputs={"domain_intent_result": domain_intent_prob, "slot_result": slot_pred_ids})
                model_tag = 'bert_e2e_nlu'
                builder.add_meta_graph_and_variables(sess,
                                                     [model_tag],
                                                     signature_def_map={
                                                         "model": signature
                                                     })
                builder.save()
        
        return output_spec
    
    return model_fn


def train_domain_intent_model(bert_config, 
                              bert_init_checkpoint, 
                              run_config, 
                              num_domain_intent_labels, 
                              num_slot_labels, 
                              num_domain_intent_lexicon,
                              num_slot_lexicon,
                              num_train_steps, 
                              num_warmup_steps, 
                              train_data_file):
    model_fn = domain_intent_model_fn_builder(
        bert_config=bert_config,
        num_domain_intent_labels=num_domain_intent_labels,
        num_slot_labels=num_slot_labels,
        num_domain_intent_lexicon=num_domain_intent_lexicon,
        num_slot_lexicon=num_slot_lexicon,
        init_checkpoint=bert_init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    # 读取record 数据，组成batch
    train_input_fn = file_based_input_fn_builder(
        input_file=train_data_file,
        seq_length=FLAGS.max_seq_length,
        feature_length=FLAGS.max_feature_length, 
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


def train_slot_model(bert_config, 
                     bert_init_checkpoint,
                     run_config, 
                     num_domain_intent_labels, 
                     num_slot_labels, 
                     num_domain_intent_lexicon,
                     num_slot_lexicon,
                     num_train_steps, 
                     num_warmup_steps, 
                     train_data_file):
    model_fn = slot_model_fn_builder(
        bert_config=bert_config,
        num_domain_intent_labels=num_domain_intent_labels,
        num_slot_labels=num_slot_labels,
        num_domain_intent_lexicon=num_domain_intent_lexicon,
        num_slot_lexicon=num_slot_lexicon,
        # 不初始化bert
        init_checkpoint=bert_init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    # 读取record 数据，组成batch
    train_input_fn = file_based_input_fn_builder(
        input_file=train_data_file,
        seq_length=FLAGS.max_seq_length,
        feature_length=FLAGS.max_feature_length, 
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    

    
def train_domain_and_intent_and_slot_model(bert_config, 
                                           bert_init_checkpoint,
                                           run_config, 
                                           num_domain_intent_labels, 
                                           num_slot_labels, 
                                           num_domain_intent_lexicon,
                                           num_slot_lexicon,
                                           num_train_steps, 
                                           num_warmup_steps, 
                                           train_data_file):
    model_fn = domain_and_intent_and_slot_model_fn_builder(
        bert_config=bert_config,
        num_domain_intent_labels=num_domain_intent_labels,
        num_slot_labels=num_slot_labels,
        num_domain_intent_lexicon=num_domain_intent_lexicon,
        num_slot_lexicon=num_slot_lexicon,
        init_checkpoint=bert_init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    # 读取record 数据，组成batch
    train_input_fn = file_based_input_fn_builder(
        input_file=train_data_file,
        seq_length=FLAGS.max_seq_length,
        feature_length=FLAGS.max_feature_length, 
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    
    
    
    
def eval_domain_and_intent_and_slot_model(bert_config, 
                                          run_config, 
                                          num_domain_intent_labels, 
                                          num_slot_labels, 
                                          num_domain_intent_lexicon,
                                          num_slot_lexicon,
                                          num_train_steps, 
                                          num_warmup_steps, 
                                          eval_data_file,
                                          num_eval_size):
    model_fn = domain_and_intent_and_slot_model_fn_builder(
        bert_config=bert_config,
        num_domain_intent_labels=num_domain_intent_labels,
        num_slot_labels=num_slot_labels,
        num_domain_intent_lexicon=num_domain_intent_lexicon,
        num_slot_lexicon=num_slot_lexicon,
        # 不初始化bert
        init_checkpoint=None,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    eval_steps = None
    if FLAGS.use_tpu:
        eval_steps = int(num_eval_size / FLAGS.eval_batch_size)
    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_data_file,
        seq_length=FLAGS.max_seq_length,
        feature_length=FLAGS.max_feature_length, 
        is_training=False,
        drop_remainder=eval_drop_remainder)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)      
    print(result)
    
    
def predict_domain_intent_slot(bert_config, 
                               run_config, 
                               num_domain_intent_labels, 
                               num_slot_labels, 
                               num_domain_intent_lexicon,
                               num_slot_lexicon,
                               num_train_steps, 
                               num_warmup_steps,
                               predict_input_fn):
    model_fn = domain_and_intent_and_slot_model_fn_builder(
        bert_config=bert_config,
        num_domain_intent_labels=num_domain_intent_labels,
        num_slot_labels=num_slot_labels,
        num_domain_intent_lexicon=num_domain_intent_lexicon,
        num_slot_lexicon=num_slot_lexicon,
        # 不初始化bert
        init_checkpoint=None,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    return estimator.predict(input_fn=predict_input_fn)


def export_model(bert_config, 
                 run_config, 
                 num_domain_intent_labels, 
                 num_slot_labels, 
                 num_domain_intent_lexicon,
                 num_slot_lexicon,
                 num_train_steps, 
                 num_warmup_steps,
                 predict_input_fn):
    
    model_fn = domain_and_intent_and_slot_model_fn_builder(
        bert_config=bert_config,
        num_domain_intent_labels=num_domain_intent_labels,
        num_slot_labels=num_slot_labels,
        num_domain_intent_lexicon=num_domain_intent_lexicon,
        num_slot_lexicon=num_slot_lexicon,
        init_checkpoint=None,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    return estimator.predict(input_fn=predict_input_fn)

    
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # 在train的时候，才删除上一轮产出的文件，在predicted的时候不做clean
    if FLAGS.clean and FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(FLAGS.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
        if os.path.exists(FLAGS.data_config_path):
            try:
                os.remove(FLAGS.data_config_path)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
                
    processor = E2eLuProcessor()

    domain_intent_labels, slot_labels, domain_intent_lexicon_names, slot_lexicon_names = processor.get_labels()
    num_domain_intent_labels = len(domain_intent_labels)
    # +1是为padding位留位置
    num_slot_labels = len(slot_labels)+1
    num_domain_intent_lexicon = len(domain_intent_lexicon_names)+1
    num_slot_lexicon = len(slot_lexicon_names) + 1
    
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path) as fd:
            data_config = json.load(fd)
    else:
        data_config = {}

    if FLAGS.do_train:
        # 从文件读取训练数据
        if len(data_config) == 0:
            train_examples = processor.get_train_examples(FLAGS.data_dir)
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

            data_config['num_train_steps'] = num_train_steps
            data_config['num_warmup_steps'] = num_warmup_steps
            data_config['num_train_size'] = len(train_examples)
        else:
            num_train_steps = int(data_config['num_train_steps'])
            num_warmup_steps = int(data_config['num_warmup_steps'])
            
        # 将数据转化为 tf_record 数据
        if data_config.get('train.tf_record_path', '') == '':
            train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
            filed_based_convert_examples_to_features(train_examples, 
                                                     domain_intent_labels, 
                                                     slot_labels, 
                                                     domain_intent_lexicon_names, 
                                                     slot_lexicon_names,
                                                     FLAGS.max_seq_length, 
                                                     FLAGS.max_feature_length, 
                                                     tokenizer, 
                                                     train_file)
        else:
            train_file = data_config.get('train.tf_record_path')
        num_train_size = num_train_size = int(data_config['num_train_size'])
        tf.logging.info("***** training data *****")
        tf.logging.info("  Num examples = %d", num_train_size)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        
    if FLAGS.do_eval:
        # 将数据转化为 tf_record 数据
        if data_config.get('eval.tf_record_path', '') == '':
            eval_examples = processor.get_dev_examples(FLAGS.data_dir)
            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            filed_based_convert_examples_to_features(eval_examples, 
                                                     domain_intent_labels, 
                                                     slot_labels, 
                                                     domain_intent_lexicon_names, 
                                                     slot_lexicon_names,
                                                     FLAGS.max_seq_length, 
                                                     FLAGS.max_feature_length, 
                                                     tokenizer, 
                                                     eval_file)
            data_config['eval.tf_record_path'] = eval_file
            data_config['num_eval_size'] = len(eval_examples)
        else:
            eval_file = data_config['eval.tf_record_path']
        num_eval_size = data_config.get('num_eval_size', 0)
        tf.logging.info("***** evaluation data *****")
        tf.logging.info("  Num examples = %d", num_eval_size)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
           
    if FLAGS.do_train: 
        # 联合训练domain,intent,slot
        tf.logging.info("***** train domain_intent and slot classification model *****")
        train_domain_and_intent_and_slot_model(bert_config,
                                               FLAGS.init_checkpoint,
                                               run_config, 
                                               num_domain_intent_labels, 
                                               num_slot_labels, 
                                               num_domain_intent_lexicon,
                                               num_slot_lexicon,
                                               num_train_steps+num_train_steps*0, 
                                               num_warmup_steps+num_train_steps*0, 
                                               train_file)


    if FLAGS.do_eval:
        eval_domain_and_intent_and_slot_model(bert_config, 
                                              run_config, 
                                              num_domain_intent_labels, 
                                              num_slot_labels, 
                                              num_domain_intent_lexicon,
                                              num_slot_lexicon,
                                              num_train_steps, 
                                              num_warmup_steps, 
                                              eval_file,
                                              num_eval_size)
        
        
    
    if FLAGS.do_predict:
        # domain vocab
        domain_intent_label_map = {}
        # index从0开始
        for (i, label) in enumerate(domain_intent_labels):
            domain_intent_label_map[i] = label
        # slot vocab
        slot_label_map = {}
        # index从1开始
        for (i, label) in enumerate(slot_labels, 1):
            slot_label_map[i] = label
        
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, domain_intent_labels, slot_labels, domain_intent_lexicon_names, slot_lexicon_names,
                                                 FLAGS.max_seq_length, FLAGS.max_feature_length, tokenizer,
                                                 predict_file, mode="test")
    
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    
        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            feature_length=FLAGS.max_feature_length, 
            is_training=False,
            drop_remainder=predict_drop_remainder)
        
        if FLAGS.do_export:
            print('export')
            pred_result = export_model(bert_config, 
                       run_config, 
                       num_domain_intent_labels, 
                       num_slot_labels, 
                       num_domain_intent_lexicon,
                       num_slot_lexicon,
                       0, 
                       0, 
                       predict_input_fn)
            for pred in pred_result:
                domain_intent_result = pred['domain_intent_result']
                slot_result = pred['slot_result']
                break
            sys.exit()
    
        pred_result = predict_domain_intent_slot(bert_config, 
                                                 run_config, 
                                                 num_domain_intent_labels, 
                                                 num_slot_labels, 
                                                 num_domain_intent_lexicon,
                                                 num_slot_lexicon,
                                                 0, 
                                                 0, 
                                                 predict_input_fn)

        
        right_domain_intent = 0
        right_slot = 0
        right = 0
        output_data = []
        for true_data, pred in zip(predict_examples, pred_result):
            domain_intent_result = pred['domain_intent_result']
            slot_result = pred['slot_result']
            
            i = np.argmax(domain_intent_result)
            pred_slot_labels = []
            for idx in range(len(true_data.text)):
                if idx >= slot_result.shape[0]:
                    continue
                slot_id = slot_result[idx]
                if slot_id == 0:
                    label = '[PAD]'
                else:
                    label = slot_label_map[slot_id]
                pred_slot_labels.append(label)
                
            domain_intent_label = domain_intent_label_map[i]
            if '-' in domain_intent_label:
                ss = domain_intent_label.split('-')
                pred_domain_label = ss[0]
                pred_intent_label = ss[1]
            if ':' in domain_intent_label:
                ss = domain_intent_label.split(':')
                pred_domain_label = ss[0]
                pred_intent_label = ss[1]
                
            if true_data.domain_intent_label == domain_intent_label:
                right_domain_intent += 1
            
            is_slot_right = True
            if not len(true_data.slot_label.split(' ')) == len(pred_slot_labels):
                is_slot_right = False
            true_slot_labels = []
            for slot_label in true_data.slot_label.split(' '):
                if slot_label == 'O':
                    true_slot_labels.append(slot_label)
                    continue
                if '-' in slot_label:
                    items = slot_label.split('-')
                    true_slot_labels.append(items[1])
                    continue
                true_slot_labels.append(slot_label)
            pred_slot_labels_simple = []
            for slot_label in pred_slot_labels:
                if slot_label == 'O':
                    pred_slot_labels_simple.append(slot_label)
                    continue
                if '-' in slot_label:
                    items = slot_label.split('-')
                    pred_slot_labels_simple.append(items[1])
                    continue
                pred_slot_labels_simple.append(slot_label)
            if not ' '.join(true_slot_labels) == ' '.join(pred_slot_labels_simple):
                is_slot_right = False
                
            if is_slot_right:
                right_slot += 1
                
            if true_data.domain_intent_label == domain_intent_label and \
                is_slot_right:
                right += 1
            
            d = {}
            d['text'] = true_data.text
            d['true_domain_intent_label'] = true_data.domain_intent_label
            d['true_slot_label'] = true_slot_labels
            d['pred_domain_intent_label'] = domain_intent_label
            d['pred_slot_label'] = pred_slot_labels_simple
            output_data.append(d)
            
        print(right_domain_intent / len(predict_examples))
        print(right_slot / len(predict_examples))
        print(right / len(predict_examples))
        
        with open(os.path.join(FLAGS.data_dir, 'pred_result.json'), 'w') as f:
            for o in output_data:
                f.write(json.dumps(o, ensure_ascii=False))
                f.write('\n')
        
    
if __name__ == "__main__":
    tf.app.run()
