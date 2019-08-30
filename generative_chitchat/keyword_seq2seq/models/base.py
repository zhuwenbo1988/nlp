#!/usr/bin/env python
# coding: utf-8


import codecs
import math
import os
import time
from abc import abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from models import model_helper
from utils import print_out, mkdir_if_not_exists, count_lines, add_summary
from models import vocab_utils as vocab


class AbstractModel(object):

    def init_embeddings(self, vocab_file, embedding_type, embedding_size, dtype=tf.float32, scope=None):
        vocab_list, vocab_size = vocab.load_vocab(vocab_file)

        with tf.variable_scope(scope or "embeddings", dtype=dtype):
            sqrt3 = math.sqrt(3)
            if embedding_type == 'random':
                print_out('# Using random embedding.')
                self.embeddings = tf.get_variable("emb_random_mat",
                                                  shape=[vocab_size, embedding_size],
                                                  initializer=tf.random_uniform_initializer(minval=-sqrt3, maxval=sqrt3, dtype=dtype))
            else:
                print_out('# Using pretrained embedding: %s.' % embedding_type)
                #TODO

    def get_keep_probs(self, mode, params):
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            encoder_keep_prob = 1.0 - params.encoder_dropout_rate
            decoder_keep_prob = 1.0 - params.decoder_dropout_rate
        else:
            encoder_keep_prob = 1.0
            decoder_keep_prob = 1.0

        return encoder_keep_prob, decoder_keep_prob

    def _get_learning_rate_decay(self, hparams, global_step, learning_rate):
        """Get learning rate decay."""
        if hparams.learning_rate_decay_scheme == "luong10":
            start_decay_step = int(hparams.num_train_steps / 2)
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 10)  # decay 10 times
            decay_factor = 0.5
        elif hparams.learning_rate_decay_scheme == "luong234":
            start_decay_step = int(hparams.num_train_steps * 2 / 3)
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 4)  # decay 4 times
            decay_factor = 0.5
        elif hparams.learning_rate_decay_scheme == "manual":
            start_decay_step = hparams.start_decay_step
            decay_steps = hparams.decay_steps
            decay_factor = hparams.decay_factor
        else:
            start_decay_step = hparams.num_train_steps
            decay_steps = 0
            decay_factor = 1.0

        print_out("  learning rate decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                      "decay_factor %g" % (hparams.learning_rate_decay_scheme,
                                           start_decay_step,
                                           decay_steps,
                                           decay_factor))

        eff_global_step = global_step
        if hparams.is_pretrain_enabled():
            eff_global_step -= hparams.num_pretrain_steps

        return tf.cond(
            eff_global_step < start_decay_step,
            lambda: learning_rate,
            lambda: tf.train.exponential_decay(
                learning_rate,
                (eff_global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def _get_sampling_probability(self, hparams, global_step, sampling_probability):
        if hparams.scheduled_sampling_decay_scheme == "luong10":
            start_decay_step = int(hparams.num_train_steps / 2)
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 10)  # decay 10 times
            decay_factor = 0.5
        elif hparams.scheduled_sampling_decay_scheme == "luong234":
            start_decay_step = int(hparams.num_train_steps * 2 / 3)
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 4)  # decay 4 times
            decay_factor = 0.5
        elif hparams.scheduled_sampling_decay_scheme == "manual":
            start_decay_step = hparams.start_decay_step
            decay_steps = hparams.decay_steps
            decay_factor = hparams.decay_factor
        else:
            start_decay_step = hparams.num_train_steps
            decay_steps = 0
            decay_factor = 1.0

        print_out("  scheduled sampling decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                      "decay_factor %g" % (hparams.scheduled_sampling_decay_scheme,
                                           start_decay_step,
                                           decay_steps,
                                           decay_factor))

        eff_global_step = global_step
        if hparams.is_pretrain_enabled():
            eff_global_step -= hparams.num_pretrain_steps

        return tf.cond(
            eff_global_step < start_decay_step,
            lambda: sampling_probability,
            lambda: tf.train.exponential_decay(
                sampling_probability,
                (eff_global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="sampling_prob_decay_cond")


class AbstractEncoderDecoderWrapper(object):
    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def interactive(self):
        pass


class NMTEncoderDecoderWrapper(AbstractEncoderDecoderWrapper):
    def __init__(self, config) -> None:
        super(NMTEncoderDecoderWrapper, self).__init__(config)
        self.config['metrics'] = ['dev_ppl']
        self.config.metrics.extend(self._get_metrics())

        for metric in self.config.metrics:
            self.config['best_{}_dir'.format(metric)] = os.path.join(self.config.model_dir, 'best_{}'.format(metric))
            mkdir_if_not_exists(self.config['best_{}_dir'.format(metric)])
            best_metric = 'best_{}'.format(metric)
            if best_metric not in self.config:
                self.config[best_metric] = float('inf')

        self.config['checkpoint_file'] = os.path.join(self.config.model_dir,
                                                      '{}.ckpt'.format(self._get_checkpoint_name()))

        if self.config.mode == 'train':
            self.config['num_train_steps'] = int(self.config.num_train_epochs * math.ceil(
                count_lines(self.config.train_data) / self.config.batch_size))

        self.config.vocab_file = os.path.join(self.config.model_dir,
                                              'vocab{}.in'.format(self.config.original_vocab_size
                                                                  if 'original_vocab_size' in self.config
                                                                  else self.config.vocab_size))

        if 'epoch_step' not in self.config:
            self.config['epoch_step'] = 0

        if 'epoch' not in self.config:
            self.config['epoch'] = 0

        self._vocab_table = None

    def _get_metrics(self):
        return []

    def _get_model_helper(self):
        raise NotImplementedError()

    def _get_checkpoint_name(self):
        raise NotImplementedError()

    def _compute_perplexity(self, model, iterator, sess, name, steps_per_log=500):
        raise NotImplementedError()

    def _consider_beam(self):
        return True

    def _pre_model_creation(self):
        raise NotImplementedError()

    def _post_model_creation(self, train_model, eval_model, infer_model):
        pass

    def _sample_decode(self,
                       model, global_step, sess, src_placeholder, batch_size_placeholder, eval_data, summary_writer):
        pass

    def _format_results(self, name, ppl, scores, metrics):
        """
        Format results.
        子类不实现这个方法
        """
        result_str = "%s ppl %.2f" % (name, ppl)
        if scores:
            for metric in metrics:
                result_str += ", %s %s %.1f" % (name, metric, scores[metric])
        return result_str

    def _get_best_results(self):
        """
        Summary of the current best results.
        子类不实现这个方法
        """
        tokens = []
        for metric in self.config.metrics:
            tokens.append("%s %.2f" % (metric, getattr(self.config, "best_" + metric)))
        return ", ".join(tokens)

    def init_stats(self):
        """Initialize statistics that we want to keep."""
        raise NotImplementedError()

    def update_stats(self, stats, summary_writer, start_time, step_result):
        """Update stats: write summary and accumulate statistics."""
        raise NotImplementedError()

    def check_stats(self, stats, global_step, steps_per_stats, log_f):
        """Print statistics and also check for overflow."""
        raise NotImplementedError()

    def _load_data(self, input_file, include_target=False):
        """Load inference data."""
        raise NotImplementedError()

    def _decode_and_evaluate(self,
                             model, infer_sess, iterator_feed_dict,
                             num_responses_per_input=1, label="test"):
        raise NotImplementedError()

    def run_internal_eval(self,
                          eval_model, eval_sess, model_dir, summary_writer, use_test_set=True):
        """Compute internal evaluation (perplexity) for both dev / test."""
        raise NotImplementedError()

    def run_full_eval(self,
                      infer_model, eval_model,
                      infer_sess, eval_sess,
                      model_dir,
                      label,
                      summary_writer):
        raise NotImplementedError()

    def run_sample_decode(self, infer_model, infer_sess, model_dir, summary_writer, eval_data):
        """Sample decode a random sentence from src_data."""
        raise NotImplementedError()


def create_attention_mechanism(attention_option, num_units, memory, memory_length):
    """Create attention mechanism based on the attention_option."""

    # Mechanism
    if attention_option == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=memory_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=memory_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=memory_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=memory_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


def create_cell(unit_type, hidden_units, num_layers, input_keep_prob=1.0, output_keep_prob=1.0, devices=None):
    if unit_type == 'lstm':
        def _new_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
    elif unit_type == 'gru':
        def _new_cell():
            return tf.contrib.rnn.GRUCell(hidden_units)
    else:
        raise ValueError('cell_type must be either lstm or gru')

    def _new_cell_wrapper(device_id=None):
        c = _new_cell()

        if input_keep_prob < 1.0 or output_keep_prob < 1.0:
            c = rnn.DropoutWrapper(c, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)

        if device_id:
            c = rnn.DeviceWrapper(c, device_id)

        return c

    if num_layers > 1:
        cells = []

        for i in range(num_layers):
            cells.append(_new_cell_wrapper(devices[i] if devices else None))

        return tf.contrib.rnn.MultiRNNCell(cells)
    else:
        return _new_cell_wrapper(devices[0] if devices else None)

