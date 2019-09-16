#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tokenization
import json
import sys
import os
import io
from collections import defaultdict


max_seq_length = 128

vocab_file = 'models/uncased_L-12_H-768_A-12/vocab.txt'
slot_vocab_file = 'semeval_aspect_ner/slot_labels.txt'

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

slot_labels = [line.strip() for line in open(slot_vocab_file)]
# slot vocab
slot_label_map = {}
# index从1开始
for (i, label) in enumerate(slot_labels, 1):
    slot_label_map[i] = label
    
def convert_single_example(query):
    global max_seq_length
    
    text = tokenization.convert_to_unicode(query)
    raw_tokens = tokenizer.tokenize(text)
    
    tokens = raw_tokens[0:(max_seq_length - 2)]
    tokens.insert(0, "[CLS]")  # 句子开始设置CLS 标志    
    tokens.append("[SEP]")  # 句尾添加[SEP] 标志
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    segment_ids = [0] * max_seq_length
    
    input_mask = [1] * len(input_ids)
    
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
    
    return input_ids, input_mask, segment_ids, raw_tokens
          

def parse_slot(tokens, labels):
    
    curr_slot = []
    slots = []
    for token, label in zip(tokens, pred_slot_labels):
        if label == 'O':
            if curr_slot:
                slots.append(' '.join(curr_slot))
            curr_slot = []
            continue
        if label == 'SINGLE-ASPECT':
            slots.append(token)
            if curr_slot:
                slots.append(' '.join(curr_slot))
            curr_slot = []
            continue
        if label == 'B-ASPECT':
            if curr_slot:
                slots.append(' '.join(curr_slot))
            curr_slot = []
            curr_slot.append(token)
            continue
        if label == 'I-ASPECT':
            curr_slot.append(token)
            continue
        if label == 'E-ASPECT':
            curr_slot.append(token)
            slots.append(' '.join(curr_slot))
            curr_slot = []
            continue
    return slots;
    
        
if __name__ == '__main__':
    
    pred_data = []
    for line in open('semeval_data/review.json'):
        o = json.loads(line)
        pred_data.append(o)
    
    model_tag = 'bert_ner'
    saved_model_dir = 'aspect_ner_model/serving_model/'
    with tf.Graph().as_default() as graph:
        sess = tf.Session()
        meta_graph_def = tf.saved_model.loader.load(sess, [model_tag], saved_model_dir)
        signature = meta_graph_def.signature_def
        input_ids_ph = graph.get_tensor_by_name(signature['model'].inputs['input_ids'].name)
        input_mask_ph = graph.get_tensor_by_name(signature['model'].inputs['input_mask'].name)
        segment_ids_ph = graph.get_tensor_by_name(signature['model'].inputs['segment_ids'].name)
    
        slot_pred_ids = graph.get_tensor_by_name(signature['model'].outputs['slot_result'].name)
    
        pred_result_list = []
        for pred_dict in pred_data:
            text = pred_dict['text']
            input_ids, input_mask, segment_ids, tokens = convert_single_example(text)

            feed_dict = {input_ids_ph: [input_ids], input_mask_ph: [input_mask], segment_ids_ph: [segment_ids]}
            pred = sess.run([slot_pred_ids], feed_dict=feed_dict)

            
            slot_result = pred[0][0]
        
            pred_slot_labels = []
            for idx in range(slot_result.shape[0]):
                slot_id = slot_result[idx]
                if slot_id == 0:
                    break
                label = slot_label_map[slot_id]
                if label in ['[CLS]', '[SEP]']:
                    continue
                pred_slot_labels.append(label)
            
            raw_slots = parse_slot(tokens, pred_slot_labels)
            slots = []
            for slot in raw_slots:
                slot = slot.replace(' ##', '')
                slot = slot.replace('##', '')
                slots.append(slot)
                
            result = {}
            result['text'] = text
            result['pred_aspect'] = slots
            pred_result_list.append(json.dumps(result, ensure_ascii=False))
            print(json.dumps(result, ensure_ascii=False))
    with open('review_pred_aspect.json') as f:
        for line in pred_result_list:
            f.write(line)
            f.write('\n')
