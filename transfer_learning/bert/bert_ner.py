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

flags.DEFINE_bool("do_export", True, "export model for inference")

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


class InputExample(object):
    """A single training/test example."""

    def __init__(self, guid, text, slot_label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """
        self.guid = guid
        self.text = text
        self.slot_label = slot_label

        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, slot_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.slot_label_ids = slot_label_ids
    
    
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

    
class NERProcessor(DataProcessor):
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
        slot_labels = []
        
        for line in open(os.path.join(FLAGS.data_dir, "slot_labels.txt"), 'r'):
            slot_labels.append(line.strip())
        
        return slot_labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            json_obj = line
            
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(json_obj['text'])
            slot_label = tokenization.convert_to_unicode(json_obj['slot_label'])
            
            examples.append(InputExample(guid=guid, text=text, slot_label=slot_label))
        return examples
    
    
def convert_single_example(ex_index, 
                             example, 
                             slot_label_list, 
                             max_seq_length, 
                             tokenizer, 
                             mode):
    # slot vocab
    slot_label_map = {}
    # index从1开始
    for (i, label) in enumerate(slot_label_list, 1):
        slot_label_map[label] = i
        
    # token -> wordpiece
    raw_tokens = example.text.split(' ')
    raw_slot_labels = example.slot_label.split(' ')
    tokens = []
    slot_labels = []
    for i, word in enumerate(raw_tokens):
        tokens.append(word)
        slot_labels.append(raw_slot_labels[i])
    # 截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        slot_labels = slot_labels[0:(max_seq_length - 2)]
    
    segment_ids = []
    slot_label_ids = []
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        segment_ids.append(0)
        slot_label_ids.append(slot_label_map[slot_labels[i]])
    segment_ids.append(0)
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
    
    # slot labels 没有CLS标志位,第一个label就是text的第一个token
    while len(slot_label_ids) < max_seq_length-1:
        slot_label_ids.append(0)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(slot_label_ids) == max_seq_length - 1
            
    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % ",".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % ",".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % ",".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % ",".join([str(x) for x in segment_ids]))
        tf.logging.info("slot_label_ids: %s" % ",".join([str(x) for x in slot_label_ids]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        slot_label_ids=slot_label_ids,
    )
    
    return feature    
    

def filed_based_convert_examples_to_features(examples, 
                                             slot_label_list, 
                                             max_seq_length, 
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
                                         slot_label_list, 
                                         max_seq_length, 
                                         tokenizer, 
                                         mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["slot_label_ids"] = create_int_feature(feature.slot_label_ids)
        
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    
    
def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "slot_label_ids": tf.FixedLenFeature([seq_length-1], tf.int64),
    }
    
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
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


def create_model(bert_config, 
                 is_training,
                 input_ids, 
                 input_mask, 
                 segment_ids, 
                 use_one_hot_embeddings,
                 input_slot_labels,
                 num_slot_labels):
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
    
    # [batch_size] 大小的向量，包含了当前batch中的序列长度
    lengths = tf.reduce_sum(used, reduction_indices=1)
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
                          is_training=is_training)
    slot_loss, slot_logits, slot_trans, slot_pred_ids = blstm_crf.add_blstm_crf_layer(crf_only=FLAGS.crf_only)
    
    return slot_loss, slot_logits, slot_trans, slot_pred_ids


def slot_model_fn_builder(bert_config, 
                                                num_slot_labels, 
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
        slot_label_ids = features["slot_label_ids"]
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        slot_loss, slot_logits, slot_trans, slot_pred_ids = create_model(bert_config, 
                                                                         is_training, 
                                                                         input_ids, 
                                                                         input_mask, 
                                                                         segment_ids, 
                                                                         use_one_hot_embeddings, 
                                                                         slot_label_ids, 
                                                                         num_slot_labels)
        with tf.variable_scope("slot_cls_op"):
            print(slot_loss)
            print(learning_rate)
            print(num_train_steps)
            print(num_warmup_steps)
            print(use_tpu)
            slot_train_op = optimization.create_optimizer(slot_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
        
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
            logging_hook = tf.train.LoggingTensorHook({"slot_loss" : slot_loss}, every_n_iter=10)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=slot_loss,
              train_op=slot_train_op,
              scaffold_fn=scaffold_fn,
              training_hooks = [logging_hook])
            
        if mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(slot_label_ids, slot_pred_ids):
                # slot
                # 首先对结果进行维特比解码
                # crf 解码
                weight = tf.sequence_mask(FLAGS.max_seq_length)
                slot_eval_prec = tf_metrics.precision(slot_label_ids, slot_pred_ids, num_slot_labels)
                slot_eval_rec = tf_metrics.recall(slot_label_ids, slot_pred_ids, num_slot_labels)
                slot_eval_f = tf_metrics.f1(slot_label_ids, slot_pred_ids, num_slot_labels)
                slot_eval_loss = tf.metrics.mean_squared_error(labels=slot_label_ids, predictions=slot_pred_ids)
                return {
                    "slot_eval_precision": slot_eval_prec,
                    "slot_eval_recall": slot_eval_rec,
                    "slot_eval_f1": slot_eval_f,
                    "slot_eval_loss": slot_eval_loss
                }
            eval_metrics = (metric_fn, [slot_label_ids, slot_pred_ids])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=slot_eval_loss,
              eval_metrics=eval_metrics,
              scaffold_fn=scaffold_fn)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            predict_result = {}
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
        
                export_dir = os.path.join(FLAGS.data_dir, 'serving_model')
        
                builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
                signature = tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs={"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}, outputs={"slot_result": slot_pred_ids})
                model_tag = 'bert_ner'
                builder.add_meta_graph_and_variables(sess,
                                                     [model_tag],
                                                     signature_def_map={
                                                         "model": signature
                                                     })
                builder.save()
        
        return output_spec
    
    return model_fn

    
def train_slot_model(bert_config, 
                     bert_init_checkpoint,
                     run_config, 
                     num_slot_labels, 
                     num_train_steps, 
                     num_warmup_steps, 
                     train_data_file):
    model_fn = slot_model_fn_builder(
        bert_config=bert_config,
        num_slot_labels=num_slot_labels,
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
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    
    
def eval_slot_model(bert_config, 
                    run_config, 
                    num_slot_labels, 
                    num_train_steps, 
                    num_warmup_steps, 
                    eval_data_file,
                    num_eval_size):
    model_fn = slot_model_fn_builder(
        bert_config=bert_config,
        num_slot_labels=num_slot_labels,
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
        is_training=False,
        drop_remainder=eval_drop_remainder)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)      
    print(result)
    
    
def predict_slot(bert_config, 
                 run_config, 
                 num_slot_labels, 
                 num_train_steps, 
                 num_warmup_steps,
                 predict_input_fn):
    model_fn = slot_model_fn_builder(
        bert_config=bert_config,
        num_slot_labels=num_slot_labels,
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
                 num_slot_labels, 
                 num_train_steps, 
                 num_warmup_steps,
                 predict_input_fn):
    
    model_fn = slot_model_fn_builder(
        bert_config=bert_config,
        num_slot_labels=num_slot_labels,
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
                
    processor = NERProcessor()

    slot_labels = processor.get_labels()
    # +1是因为index=0是padding位
    num_slot_labels = len(slot_labels)+1

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
            filed_based_convert_examples_to_features(
                train_examples, slot_labels, FLAGS.max_seq_length, tokenizer, train_file)
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
            filed_based_convert_examples_to_features(
                eval_examples, slot_labels, FLAGS.max_seq_length, tokenizer, eval_file)
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
        train_slot_model(bert_config,
                         FLAGS.init_checkpoint,
                         run_config, 
                         num_slot_labels, 
                         num_train_steps, 
                         num_warmup_steps, 
                         train_file)

    if FLAGS.do_eval:
        eval_slot_model(bert_config, 
                        run_config, 
                        num_slot_labels, 
                        num_train_steps, 
                        num_warmup_steps, 
                        eval_file,
                        num_eval_size)
    
    if FLAGS.do_predict:
        # slot vocab
        slot_label_map = {}
        # index从1开始
        for (i, label) in enumerate(slot_labels, 1):
            slot_label_map[i] = label
        
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, slot_labels,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, mode="test")
    
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    
        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)
        
        if FLAGS.do_export:
            print('export')
            pred_result = export_model(bert_config, 
                       run_config, 
                       num_slot_labels, 
                       0, 
                       0, 
                       predict_input_fn)
            for pred in pred_result:
                slot_result = pred['slot_result']
                break
            sys.exit()
    
        # TODO
        pred_result = predict_slot(bert_config, 
                                   run_config, 
                                   num_slot_labels, 
                                   0, 
                                   0, 
                                   predict_input_fn)
        
                
if __name__ == "__main__":
    tf.app.run()
