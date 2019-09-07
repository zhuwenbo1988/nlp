# coding=utf-8

import modeling
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.tools.graph_transforms import TransformGraph
from run_classifier import create_model
import os

input_model_checkpoint = ''
export_model_dir = ''
export_model_name = ''

bert_config_file = 'models/uncased_L-12_H-768_A-12/bert_config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)

label_file = os.path.join(input_model_checkpoint, 'labels.txt')
label_list = []
for line in open(label_file):
    label = line.strip()
    label_list.append(label)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
global graph
graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    is_training = False
    use_one_hot_embeddings = False
    batch_size = 1
    num_labels = len(label_list)
    input_ids_p = tf.placeholder(tf.int32, [batch_size, None], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, None], name="input_mask")
    label_ids_p = tf.placeholder(tf.int32, [batch_size], name="label_ids")
    segment_ids_p = tf.placeholder(tf.int32, [None], name="segment_ids")
    total_loss, per_example_loss, logits, probabilities = create_model(
        bert_config, is_training, input_ids_p, input_mask_p, segment_ids_p,
        label_ids_p, num_labels, use_one_hot_embeddings)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(input_model_checkpoint, 'model_output')))
    graph_def = tf.get_default_graph().as_graph_def()

    # 1
    # variables + saved_model.pb
    tf.saved_model.simple_save(sess,
            export_model_dir,
            inputs={"input_ids": input_ids_p,
                    "input_mask": input_mask_p,
                    "label_ids": label_ids_p,
                    "segment_ids": segment_ids_p},
            outputs={"probabilities": probabilities})

    # 2
    # only .pb
    input_node_names = ['input_ids', 'input_mask', 'label_ids', 'segment_ids']
    output_node_names = ['loss/probabilities']

    transforms = [
        'remove_nodes(op=Identity)',
        'fold_constants(ignore_errors=true)',
        'fold_batch_norms',
        'merge_duplicate_nodes',
        'strip_unused_nodes',
        'sort_by_execution_order'
    ]

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['loss/probabilities', 'input_ids', 'input_mask', 'label_ids', 'segment_ids'])
    output_graph_def = TransformGraph(
            output_graph_def,
            input_node_names,
            output_node_names,
            transforms
        )

    save_path = os.path.join(export_model_dir, export_model_name + ".pb")
    with tf.gfile.FastGFile(save_path, mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
        print("froze graph save to path: ", save_path)
