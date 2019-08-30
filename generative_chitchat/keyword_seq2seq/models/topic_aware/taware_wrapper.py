#!/usr/bin/env python
# coding: utf-8


import math
import os
import random
import time
import codecs

import tensorflow as tf

from models import model_helper
from models.base import NMTEncoderDecoderWrapper
from models.topic_aware import taware_helper
from models.topic_aware import taware_model
from models import vocab_utils as vocab
from utils import split3, print_out, add_summary, safe_exp, Stopwatch, print_time, rm_if_exists


class TopicAwareNMTEncoderDecoder(NMTEncoderDecoderWrapper):
    def __init__(self, config) -> None:
        if config.mode != 'train' and 'original_vocab_size' in config:
            config.vocab_size = config.original_vocab_size

        super(TopicAwareNMTEncoderDecoder, self).__init__(config)

    def _get_checkpoint_name(self):
        return 'taware'

    def _pre_model_creation(self):
        self.config['topic_vocab_file'] = os.path.join(split3(self.config.vocab_file)[0], 'topic_vocab.in')
        # 创建vocab文件
        self._vocab_table, self.__topic_vocab_table = vocab.initialize_vocabulary(self.config)

        if 'original_vocab_size' not in self.config:
            self.config['original_vocab_size'] = self.config.vocab_size
        self.config.vocab_size = len(self._vocab_table)
        self.config.topic_vocab_size = len(self.__topic_vocab_table)

    def run_sample_decode(self, infer_model, infer_sess, model_dir, summary_writer, eval_data):
        """Sample decode a random sentence from src_data."""
        print('loading infer model')
        with infer_model.graph.as_default():
            loaded_infer_model, global_step = model_helper.create_or_load_model(
                infer_model.model, model_dir, infer_sess, "infer")

        self.__sample_decode(loaded_infer_model, global_step, infer_sess,
                            infer_model.iterator, eval_data,
                            infer_model.src_placeholder,
                            infer_model.batch_size_placeholder, summary_writer)

    def run_internal_eval(self,
                          eval_model, eval_sess, model_dir, summary_writer, use_test_set=True):
        """Compute internal evaluation (perplexity) for both dev / test."""
        with eval_model.graph.as_default():
            loaded_eval_model, global_step = model_helper.create_or_load_model(
                eval_model.model, model_dir, eval_sess, "eval")

        dev_file = self.config.dev_data

        dev_eval_iterator_feed_dict = {
            eval_model.eval_file_placeholder: dev_file
        }

        print_out("eval dev set")
        dev_ppl = self._internal_eval(loaded_eval_model, global_step, eval_sess,
                                      eval_model.iterator, dev_eval_iterator_feed_dict,
                                      summary_writer, "dev")
        add_summary(summary_writer, global_step, "dev_ppl", dev_ppl)

        if dev_ppl < self.config.best_dev_ppl:
            loaded_eval_model.saver.save(eval_sess,
                                         os.path.join(self.config.best_dev_ppl_dir, 'taware.ckpt'),
                                         global_step=global_step)

        test_ppl = None
        if use_test_set:
            test_file = self.config.test_data

            test_eval_iterator_feed_dict = {
                eval_model.eval_file_placeholder: test_file
            }
            print_out("eval test set")
            test_ppl = self._internal_eval(loaded_eval_model, global_step, eval_sess,
                                           eval_model.iterator, test_eval_iterator_feed_dict,
                                           summary_writer, "test")
        return dev_ppl, test_ppl

    def init_stats(self):
        """Initialize statistics that we want to keep."""
        return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0,
                "total_count": 0.0, "grad_norm": 0.0}

    def update_stats(self, stats, summary_writer, start_time, step_result):
        """Update stats: write summary and accumulate statistics."""
        (_, step_loss, step_predict_count, step_summary, global_step,
         step_word_count, batch_size, grad_norm, learning_rate) = step_result

        # Write step summary.
        summary_writer.add_summary(step_summary, global_step)

        # update statistics
        stats["step_time"] += (time.time() - start_time)
        stats["loss"] += (step_loss * batch_size)
        stats["predict_count"] += step_predict_count
        stats["total_count"] += float(step_word_count)
        stats["grad_norm"] += grad_norm
        stats["learning_rate"] = learning_rate

        return global_step

    def check_stats(self, stats, global_step, steps_per_stats, log_f):
        """Print statistics and also check for overflow."""
        # Print statistics for the previous epoch.
        avg_step_time = stats["step_time"] / steps_per_stats
        avg_grad_norm = stats["grad_norm"] / steps_per_stats
        train_ppl = safe_exp(
            stats["loss"] / stats["predict_count"])
        speed = stats["total_count"] / (1000 * stats["step_time"])
        print_out(
            "  global step %d lr %g "
            "step-time %.2fs wps %.2fK ppl %.2f gN %.2f %s" %
            (global_step, stats["learning_rate"],
             avg_step_time, speed, train_ppl, avg_grad_norm,
             self._get_best_results()),
            log_f)

        # Check for overflow
        is_overflow = False
        if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
            print_out("  step %d overflow, stop early" % global_step, log_f)
            is_overflow = True

        return train_ppl, speed, is_overflow

    def _load_data(self, input_file, include_target=False):
        """Load inference data."""

        inference_data = []
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(input_file, mode="rb")) as f:
            for line in f:
                utterances_str, topics_str = tuple(line.split(vocab.DEFAULT_SEPARATOR))
                utterances, topics = utterances_str.strip().split('\t'), topics_str.strip().split()
                inference_data.append((utterances, topics))

        return inference_data

    def train(self, target_session="", scope=None):
        out_dir = self.config.model_dir
        model_dir = out_dir

        num_train_steps = self.config.num_train_steps
        steps_per_stats = self.config.steps_per_stats
        steps_per_eval = self.config.steps_per_eval
        if not steps_per_eval:
            steps_per_eval = 20 * steps_per_stats

        # 创建词表
        self._pre_model_creation()

        print_out("model scope: %s" % scope)
        train_model = taware_helper.create_train_model(taware_model.TopicAwareSeq2SeqModel, self.config, scope)
        eval_model = taware_helper.create_eval_model(taware_model.TopicAwareSeq2SeqModel, self.config, scope)
        infer_model = taware_helper.create_infer_model(taware_model.TopicAwareSeq2SeqModel, self.config, scope)

        # Preload data for sample decoding.
        dev_file = self.config.dev_data
        eval_data = self._load_data(dev_file, include_target=True)

        summary_name = "train_log"

        # Log and output files
        log_file = os.path.join(out_dir, "log_%d" % time.time())
        log_f = tf.gfile.GFile(log_file, mode="a")
        print_out("# log_file=%s" % log_file, log_f)

        avg_step_time = 0.0

        # TensorFlow model
        config_proto = model_helper.get_config_proto(self.config.log_device)

        train_sess = tf.Session(
            target=target_session, config=config_proto, graph=train_model.graph)
        eval_sess = tf.Session(
            target=target_session, config=config_proto, graph=eval_model.graph)
        infer_sess = tf.Session(
            target=target_session, config=config_proto, graph=infer_model.graph)

        with train_model.graph.as_default():
            loaded_train_model, global_step = model_helper.create_or_load_model(
                train_model.model, model_dir, train_sess, "train")

        # Summary writer
        summary_writer = tf.summary.FileWriter(
            os.path.join(out_dir, summary_name), train_model.graph)

        last_stats_step = global_step
        last_eval_step = global_step
        # last_external_eval_step = global_step
        patience = self.config.patience

        # This is the training loop.
        stats = self.init_stats()
        speed, train_ppl = 0.0, 0.0
        start_train_time = time.time()

        print_out(
            "# Start step %d, epoch %d, lr %g, %s" %
            (global_step, self.config.epoch, loaded_train_model.learning_rate.eval(session=train_sess),
             time.ctime()),
            log_f)

        self.config.save()
        print_out("# Configs saved")

        # Initialize all of the iterators
        skip_count = self.config.batch_size * self.config.epoch_step
        print_out("# Init train iterator for %d steps, skipping %d elements" %
                      (self.config.num_train_steps, skip_count))

        train_sess.run(
            train_model.iterator.initializer,
            feed_dict={train_model.skip_count_placeholder: skip_count})

        while self.config.epoch < self.config.num_train_epochs and patience > 0:
            ### Run a step ###
            start_time = time.time()
            try:
                step_result = loaded_train_model.train(train_sess)
                self.config.epoch_step += 1
            except tf.errors.OutOfRangeError:
                # Finished going through the training dataset.  Go to next epoch.
                sw = Stopwatch()
                print_out(
                    "# Finished an epoch, step %d. Perform external evaluation" %
                    global_step)
                self.run_sample_decode(infer_model, infer_sess,
                                       model_dir, summary_writer, eval_data)

                print_out(
                    "## Done epoch %d in %d steps. step %d @ eval time: %ds" %
                    (self.config.epoch, self.config.epoch_step, global_step, sw.elapsed()))

                self.config.epoch += 1
                self.config.epoch_step = 0
                self.config.save()

                train_sess.run(
                    train_model.iterator.initializer,
                    feed_dict={train_model.skip_count_placeholder: 0})
                continue

            # Write step summary and accumulate statistics
            global_step = self.update_stats(stats, summary_writer, start_time, step_result)

            # Once in a while, we print statistics.
            if global_step - last_stats_step >= steps_per_stats:
                last_stats_step = global_step
                train_ppl, speed, is_overflow = self.check_stats(stats, global_step, steps_per_stats, log_f)
                if is_overflow:
                    break

                # Reset statistics
                stats = self.init_stats()

            if global_step - last_eval_step >= steps_per_eval:
                last_eval_step = global_step

                print_out("# Save eval, global step %d" % global_step)
                add_summary(summary_writer, global_step, "train_ppl", train_ppl)

                # Save checkpoint
                loaded_train_model.saver.save(
                    train_sess,
                    self.config.checkpoint_file,
                    global_step=global_step)

                # Evaluate on dev
                self.run_sample_decode(infer_model, infer_sess, model_dir, summary_writer, eval_data)
                dev_ppl, _ = self.run_internal_eval(eval_model, eval_sess, model_dir, summary_writer, use_test_set=False)

                # 当patience=0,则提前结束训练
                if dev_ppl < self.config.best_dev_ppl:
                    self.config.best_dev_ppl = dev_ppl
                    patience = self.config.patience
                    print_out('    ** Best model thus far, ep {}|{} dev_ppl {:.3f}'.format(
                        self.config.epoch,
                        self.config.epoch_step,
                        dev_ppl))
                elif dev_ppl > self.config.degrade_threshold * self.config.best_dev_ppl:
                    patience -= 1
                    print_out(
                        '    worsened, ep {}|{} patience {} best_dev_ppl {:.3f}'.format(
                            self.config.epoch,
                            self.config.epoch_step,
                            self.config.patience,
                            self.config.best_dev_ppl))

                # Save config parameters
                self.config.save()

        # Done training
        loaded_train_model.saver.save(
            train_sess,
            self.config.checkpoint_file,
            global_step=global_step)

        dev_scores, test_scores, dev_ppl, test_ppl = None, None, None, None
        result_summary = ""

        print_out(
            "# Final, step %d lr %g "
            "step-time %.2f wps %.2fK ppl %.2f, %s, %s" %
            (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
             avg_step_time, speed, train_ppl, result_summary, time.ctime()),
            log_f)
        print_time("# Done training!", start_train_time)

        summary_writer.close()

        return (dev_scores, test_scores, dev_ppl, test_ppl, global_step)

    def __sample_decode(self,
                       model, global_step, sess, iterator, eval_data,
                       iterator_src_placeholder,
                       iterator_batch_size_placeholder, summary_writer):
        """Pick a sentence and decode."""
        decode_id = random.randint(0, len(eval_data) - 1)
        print_out("  # {}".format(decode_id))

        sample_utterances, sample_topic_words = eval_data[decode_id]
        sample_topic_words = " ".join(sample_topic_words)

        iterator_feed_dict = {
            iterator_src_placeholder: [vocab.DEFAULT_SEPARATOR.join(["\t".join(sample_utterances), sample_topic_words])],
            iterator_batch_size_placeholder: 1,
        }
        sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

        ncm_outputs, attention_summary = model.decode(sess)

        if self.config.beam_width > 0:
            # get the top translation.
            ncm_outputs = ncm_outputs[0]

        translation = vocab.get_translation(ncm_outputs, sent_id=0)
        print_out("    sources:")
        for t, src in enumerate(sample_utterances[:-1]):
            print_out("      @{} {}".format(t + 1, src))
        print_out("    topicals: {}".format(sample_topic_words))
        print_out("    resp: {}".format(sample_utterances[-1]))
        print_out(b"    ncm: " + translation)

        # Summary
        if attention_summary is not None:
            summary_writer.add_summary(attention_summary, global_step)

    def _internal_eval(self,
                       model, global_step, sess, iterator, iterator_feed_dict,
                       summary_writer, label):
        """Computing perplexity."""
        sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
        ppl = model_helper.compute_perplexity(model, sess, label)
        add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
        return ppl

    def test(self):
        from prompt_toolkit import prompt

        self._pre_model_creation()

        infer_model = taware_helper.create_infer_model(taware_model.TopicAwareSeq2SeqModel, self.config)
        config_proto = model_helper.get_config_proto(self.config.log_device)

        with tf.Session(graph=infer_model.graph, config=config_proto) as sess:
            ckpt = tf.train.latest_checkpoint(self.config.model_dir)
            loaded_infer_model = model_helper.load_model(
                infer_model.model, ckpt, sess, "infer")

            print_out("# Start decoding")

            input_file = self.config.test_data
            inference_data = []
            with codecs.getreader("utf-8")(tf.gfile.GFile(input_file, mode="rb")) as f:
                for line in f:
                    line = line.strip()
                    inference_data.append(line)

            for inp in inference_data:
                if 'null' in inp:
                    print(inp)
                    continue
                ss = inp.split('|')
                sss = ss[0].strip().split('\t')
                q = sss[0]
                #w = sss[1]
                words = ss[1].strip()
                inp = '{}  |  {}'.format(q, words)         
                iterator_feed_dict = {
                    infer_model.src_placeholder: [inp],
                    infer_model.batch_size_placeholder: 1,
                }
                sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
                output, _ = loaded_infer_model.decode(sess)

                if self.config.beam_width > 0:
                    # get the top translation.
                    output = output[0]

                resp = vocab.get_translation(output, sent_id=0)
                print('{}\t{}'.format(q, resp.decode('utf-8')))
    def create_infer_model_graph(self):
        self._pre_model_creation()
        infer_model = taware_helper.create_infer_model(taware_model.TopicAwareSeq2SeqModel, self.config)
        return infer_model