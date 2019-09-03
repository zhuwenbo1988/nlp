# coding=utf-8

import tensorflow as tf
import data_helper as dh
import numpy as np
import os
import math
import sys
import configparser
import sse_model
import shutil


conf_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(conf_file)

# 模型的保存位置
model_dir = config.get('train', 'model_loc')

# 输入向量的长度
vec_size = config.get('vector', 'vector_size')
# 输出向量的长度
out_size = config.get('train', 'reshape_size')

learning_rate = float(config.get('train', 'learning_rate'))
loss_weight = float(config.get('train', 'loss_func_weight'))
batch_size = int(config.get('train', 'batch_size'))
max_epoc = int(config.get('train', 'epoch_nums'))
steps_per_checkpoint = int(config.get('train', 'steps_per_checkpoint'))

dev_percent = float(config.get('train', 'dev_percent'))
train_data, dev_data, train_size, dev_size, all_vector = dh.load_data(config, dev_percent)

epoc_steps = math.ceil(int(train_size/batch_size))

print('train size {}'.format(train_size))
print('dev size {}'.format(dev_size))
print('steps ever epoch {}'.format(epoc_steps))
print('vector nums {}'.format(len(all_vector)))

def batch_iter(data, batch_size, shuffle=True):
  data = np.asarray(data)
  data_size = len(data)
  num_batches_per_epoch = math.ceil(int(data_size/batch_size))

  if shuffle:
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
  else:
    shuffled_data = data

  for batch_num in range(num_batches_per_epoch):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, data_size)
    yield shuffled_data[start_index:end_index]

cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
cfg.gpu_options.allow_growth = True

with tf.Session(config=cfg) as sess:

  print('init model')
  sse = sse_model.SSE(vec_size, out_size, learning_rate, loss_weight)
  for var in tf.trainable_variables():
    print(var)

  def save_serving_model(save_parent_dir, model):
    """
    这个模型会再被导出为QAP要求的模型
    """
    saved_model_dir = os.path.join(save_parent_dir, 'serving_model')
    if os.path.exists(saved_model_dir):
      shutil.rmtree(saved_model_dir)
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'src_in': model.src_input, "tgt_in": model.tgt_input},
        outputs={'sim': model.predict_similarity})
    model_tag = 'sse'
    builder.add_meta_graph_and_variables(sess,
                                         [model_tag],
                                         signature_def_map={
                                             "model": signature
                                         })
    builder.save()

  # 保存实验用的模型
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
  checkpoint_path = os.path.join(model_dir, "SSE.ckpt")

  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    last_dev_acc = float(ckpt.model_checkpoint_path.split('-')[-2])
    print('last best accuracy {}'.format(last_dev_acc))
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    last_dev_acc = 0
    sess.run(tf.global_variables_initializer())

  loss = tf.summary.scalar("loss (raw)", sse.loss)
  summary_op = tf.summary.merge([loss])
  sw =  tf.summary.FileWriter(logdir=model_dir,  graph=sess.graph, flush_secs=120)

  # 统计每个观测点的loss和acc
  loss, train_acc = 0.0, 0.0
  current_step = 0
  for epoch in range(max_epoc):
    print('epoch {}'.format(epoch))
    
    train_iter = batch_iter(list(zip(train_data[0], train_data[1], train_data[2])), batch_size)

    for batchId in range(int(epoc_steps)):

      batch = next(train_iter)
      src_inputs, tgt_inputs, labels = zip(*batch)

      feed_dict = {}
      # 使用这种方式获取向量可以节约空间
      feed_dict[sse.src_input] = all_vector[np.array(src_inputs)]
      feed_dict[sse.tgt_input] = all_vector[np.array(tgt_inputs)]
      feed_dict[sse.labels] = labels

      train_ops = [sse.train, summary_op, sse.loss, sse.train_acc]

      _, summary, step_loss, step_train_acc = sess.run(train_ops, feed_dict=feed_dict)

      sw.add_summary(summary, current_step)

      loss += step_loss / steps_per_checkpoint
      train_acc += step_train_acc / steps_per_checkpoint

      current_step += 1
      if current_step % steps_per_checkpoint == 0:
        acc_sum = tf.Summary(value=[tf.Summary.Value(tag="train_acc", simple_value=train_acc)])
        sw.add_summary(acc_sum, current_step)

        print('current train accuracy {} loss {} step {}'.format(train_acc, loss, current_step))

        loss, train_acc = 0.0, 0.0

    dev_iter = batch_iter(list(zip(dev_data[0], dev_data[1], dev_data[2])), batch_size, shuffle=False)
    dev_steps = math.ceil(int(dev_size/batch_size))    
    dev_acc = 0.0
    for n_step in range(int(dev_steps)):
      batch = next(dev_iter)
      src_inputs, tgt_inputs, labels = zip(*batch)

      feed_dict = {}
      feed_dict[sse.src_input] = all_vector[np.array(src_inputs)]
      feed_dict[sse.tgt_input] = all_vector[np.array(tgt_inputs)]
      feed_dict[sse.labels] = labels

      dev_ops = [sse.train_acc]

      step_dev_acc = sess.run(dev_ops, feed_dict=feed_dict)
      dev_acc += step_dev_acc[0] / dev_steps

    print('epoch {} dev accuracy {}'.format(epoch, dev_acc))
    # 保存最好的模型
    if dev_acc >= last_dev_acc:
      last_dev_acc = dev_acc
      current_step = tf.train.global_step(sess, sse.global_step)
      saver.save(sess, '{}-epoch-{}-dev_acc-{}'.format(checkpoint_path, epoch, dev_acc), current_step)
      save_serving_model(model_dir, sse)
