# coding=utf-8

import tokenization
import json
import re


vocab_file = 'models/uncased_L-12_H-768_A-12/vocab.txt'
do_lower_case = True
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def split(s):
  s = tokenization.convert_to_unicode(s)
  tokens = tokenizer.tokenize(s)
  return tokens


def get_ner_label(words):
  size = len(words)
  if size == 1:
    return 'SINGLE-ASPECT'
  labels = []
  for i, w in enumerate(words):
    if i == 0:
      labels.append('B-ASPECT')
      continue
    if i == size-1:
      labels.append('E-ASPECT')
      continue
    labels.append('I-ASPECT')
  return ' '.join(labels)


d = {}
d['O'] = 1
d['SINGLE-ASPECT'] = 1
d['B-ASPECT'] = 1
d['E-ASPECT'] = 1
d['I-ASPECT'] = 1


# or semeval_aspect_ner/restaurant_review.json
f = 'semeval_aspect_ner/laptop_review.json'
for line in open(f):
  o = json.loads(line)
  text = o['text']

  s = tokenization.convert_to_unicode(text)
  tokens = tokenizer.tokenize(s)
  ss = ' '.join(tokens)
  bad = False
  if 'aspect' in o:
    for a in o['aspect']:
      aspect_value = a[0]
      a_tokens = split(aspect_value)
      a_s = ' '.join(a_tokens)
      if a_s not in ss:
        bad = True
        continue
      label = get_ner_label(a_tokens)
      ss = ss.replace(a_s, label)
  if bad:
    continue
  n_tokens = ss.split()
  labels = []
  for w in n_tokens:
    if 'ASPECT' not in w:
      labels.append('O')
    else:
      labels.append(w)
  for label in labels:
    if label not in d:
      bad = True
  if bad:
    continue
  res = {}
  res['text'] = ' '.join(tokens)
  res['slot_label'] = ' '.join(labels)
  print(json.dumps(res, ensure_ascii=False))
