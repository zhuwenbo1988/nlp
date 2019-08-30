# coding=utf-8

import data_helper
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from itertools import cycle

from config import GlobalConfig
from text_cnn import *


cf = GlobalConfig()


def print_result(y_prediction_list, probs, y_true_list, vocab, label_vocab):
    writer = pd.ExcelWriter("./result/test_result.xlsx")
    xls = pd.ExcelFile(filename)
    sheet1 = xls.parse("sentiment")
    # 读取测试的句子
    queries = sheet1["query"]

    # 全部输出
    y_prediction = [label_vocab[x] for x in y_prediction_list]
    y_score = [ps[idx] for idx, ps in zip(y_prediction_list, probs)]
    y_true = [label_vocab[x] for x in y_true_list]

    # 只输出badcase
    # y_prediction, y_score, y_true, queries = zip(*[(label_vocab[yp], ps[yp], label_vocab[yt], q) for yp, ps, yt, q in zip(y_prediction_list, probs, y_true_list, queries) if yp != yt])

    df = {'query': pd.Series(queries), 'predict': pd.Series(y_prediction), 'label': pd.Series(y_true), 'score':pd.Series(y_score)}
    df = pd.DataFrame(df, columns=["query", "predict", "label", "score"], index=None)
    df.to_excel(writer, 'sentiment', index=False)
    writer.save()


def test(model_path,
         x_dev,
         y_dev,
         sents_dev,
         patt_dev):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=cf.allow_soft_placement,
            log_device_placement=cf.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():

            tf.saved_model.loader.load(sess, ['training_model'], model_path)

            feed_dict = {
                'input_x:0': x_dev,
                'labels:0': y_dev,
                'sentiment:0': sents_dev,
                'patterns:0': patt_dev,
                'dropout_keep_prob:0': 1.0}
            predictions, probs = sess.run(['output/predictions:0', 'output/logits:0'], feed_dict=feed_dict)
            print("classification_report: ")
            print(classification_report(y_dev, predictions))
            print("accuracy_score: ")
            print(accuracy_score(y_dev, predictions))

            y_one_hot = label_binarize(y_dev, np.arange(3))
            # 1、调用函数计算micro类型的AUC
            print '调用函数auc：', roc_auc_score(y_one_hot, probs, average='micro')
            # 2、手动计算micro类型的AUC
            #首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
            fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), probs.ravel())
            auc_val = auc(fpr, tpr)
            print '手动计算auc：', auc_val
            #绘图
            mpl.rcParams['font.sans-serif'] = u'SimHei'
            mpl.rcParams['axes.unicode_minus'] = False
            #FPR就是横坐标,TPR就是纵坐标
            plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc_val)
            plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
            plt.xlim((-0.01, 1.02))
            plt.ylim((-0.01, 1.02))
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.xlabel('False Positive Rate', fontsize=13)
            plt.ylabel('True Positive Rate', fontsize=13)
            plt.grid(b=True, ls=':')
            plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
            plt.title(u'sentiment classification', fontsize=17)
            plt.show()

            # 输出结果
            print_result(predictions, probs, y_dev, vocab, label_vocab)


if __name__ == '__main__':
    #
    test_data_file = sys.argv[1]
    export_model = sys.argv[2]

    # load data
    print("Loading data...")
    filename = os.path.join(cf.data_path, test_data_file)
    X, Y, sentiment_dict_features, patterns = data_helper.map_file_to_ids(filename=filename)
    # 输出使用
    _, vocab, _, label_vocab = data_helper.load_vocab(filename)

    # random shuffle data
    np.random.seed(1)
    shuffle_indices = np.random.permutation(len(X))
    X_shuffled = X
    Y_shuffled = Y

    test(os.path.join(cf.model_path, export_model), X, Y, sentiment_dict_features, patterns)
