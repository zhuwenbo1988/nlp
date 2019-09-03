# coding=utf-8

import sys
import configparser
import os
import json
from collections import defaultdict
from random import shuffle


def parse(config):
  data_path = config.get('preprocess', 'data_loc')
  raw_data_file = os.path.join(data_path, config.get('preprocess', 'raw_data_name'))
  text_data_file = os.path.join(data_path, config.get('preprocess', 'text_data_name'))
  with open(text_data_file, 'w') as f:
    for line in open(raw_data_file):
      line = line.strip()
      JSON = json.loads(line)
      for s in JSON['paraphrase']:
        f.write(s)
        f.write('\n')

def map_index(config):
  data_path = config.get('preprocess', 'data_loc')
  raw_data_file = os.path.join(data_path, config.get('preprocess', 'raw_data_name'))
  index_file = os.path.join(data_path, config.get('preprocess', 'index_data_name'))
  with open(index_file, 'w') as f:
    line_id = 0
    for ques_id, line in enumerate(open(raw_data_file)):
      line = line.strip()
      JSON = json.loads(line)
      for s in JSON['paraphrase']:
        f.write('{}\t{}\t{}\n'.format(s, ques_id, line_id))
        line_id += 1

def map_relation(config):
  data_path = config.get('preprocess', 'data_loc')
  index_file = os.path.join(data_path, config.get('preprocess', 'index_data_name'))
  rel_file = os.path.join(data_path, config.get('preprocess', 'relation_data_name'))

  tmp = defaultdict(list)
  for line in open(index_file):
    line = line.strip()
    items = line.split('\t')
    gid = items[1]
    qid = items[2]
    tmp[gid].append(qid)

  train_rel = []
  for gid_1 in tmp:
    for id_1 in tmp[gid_1]:
      pos = []
      for id_2 in tmp[gid_1]:
        if id_1 == id_2:
          continue
        pos.append((id_1, id_2, 1))

      neg = []
      for gid_2 in tmp:
        if gid_1 == gid_2:
          continue
        ids = tmp[gid_2]
        neg.extend([(id_1, id_3, 0) for id_3 in ids])
      train_rel.extend(pos)
      train_rel.extend(neg)

  shuffle(train_rel)
  with open(rel_file, 'w') as f:
    for id_1, id_2, label in train_rel:
      f.write('{}\t{}\t{}\n'.format(id_1, id_2, label))

if __name__ == '__main__':
  if not len(sys.argv) == 3:
    print('need function name and preprocess config file')
    sys.exit()

  conf_file = sys.argv[1]
  config = configparser.ConfigParser()
  config.read(conf_file)

  func_name = sys.argv[2]
  func = globals()[func_name]
  func(config)
