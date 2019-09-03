# coding=utf-8

import numpy as np
import os


def _load_vector(config):
  """
  加载向量文件
  """
  data_path = config.get('preprocess', 'data_loc')
  vec_file = os.path.join(data_path, config.get('vector', 'vector_file'))

  vec = []
  for line in open(vec_file):
    line = line.strip()
    vec.append(eval(line))

  return np.array(vec)

def _load_relation(config):
  """
  加载relation pair文件
  """
  data_path = config.get('preprocess', 'data_loc')
  rel_file = os.path.join(data_path, config.get('preprocess', 'relation_data_name'))

  src_vec = []
  tgt_vec = []
  label_vec = []
  for line in open(rel_file):
    line = line.strip()
    items = line.split('\t')
    src_id = int(items[0])
    tgt_id = int(items[1])
    label = int(items[2])
    src_vec.append(src_id)
    tgt_vec.append(tgt_id)
    label_vec.append(label)

  return np.array(src_vec), np.array(tgt_vec), np.array(label_vec)

def load_data(config, dev_percent=0.2):
  print('loading relation')
  src_vec, tgt_vec, label_vec = _load_relation(config)

  print('shuffle data')
  data_size = label_vec.shape[0]
  shuffle_indices = np.random.permutation(np.arange(data_size))
  shuffled_src = src_vec[shuffle_indices]
  shuffled_tgt = tgt_vec[shuffle_indices]
  shuffled_label = label_vec[shuffle_indices]

  print('split data')
  dev_idx = int(-1*data_size*dev_percent)
  train_src, dev_src = shuffled_src[:dev_idx], shuffled_src[dev_idx:]
  train_tgt, dev_tgt = shuffled_tgt[:dev_idx], shuffled_tgt[dev_idx:]
  train_label, dev_label = shuffled_label[:dev_idx], shuffled_label[dev_idx:]

  train_data = (train_src, train_tgt, train_label)
  train_size = train_label.shape[0]
  dev_data = (dev_src, dev_tgt, dev_label)
  dev_size = dev_label.shape[0]

  print('loading vector')
  all_vector = _load_vector(config)

  return train_data, dev_data, train_size, dev_size, all_vector
