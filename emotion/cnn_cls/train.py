# coding=utf-8

import datetime
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

import data_helper
from config import GlobalConfig
from text_cnn import *


cf = GlobalConfig()


def print_result(x_list, y_prediction_list, y_true_list, ivocab, label_ivocab, i):
    writer = pd.ExcelWriter("./result/result_" + str(i) + ".xlsx")
    length = len(x_list)
    x_strs = []
    for i in range(length):
        x = x_list[i]
        x_str = ""
        for w in x:
            if w != 0:
                x_str += ivocab[w]
            else:
                continue
        x_strs.append(x_str)

    y_prediction = [label_ivocab[x] for x in y_prediction_list]
    y_true = [label_ivocab[x] for x in y_true_list]
    df = {'query': pd.Series(x_strs), 'predict': pd.Series(
        y_prediction), 'label': pd.Series(y_true)}
    df = pd.DataFrame(df, columns=["query", "predict", "label"], index=None)
    df.to_excel(writer, 'sentiment', index=False)
    writer.save()


def train(x_train,
          y_train,
          # 情感词典
          sents_train,
          patt_train,
          x_dev,
          y_dev,
          sents_dev,
          patt_dev,
          idx):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=cf.allow_soft_placement,
            log_device_placement=cf.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=cf.max_sequence_length,
                sentiment_length=cf.max_sentiment_len,
                patterns_length=cf.max_pattern_length,
                num_classes=num_classes,
                vocab_size=vocab_size,
                embedding_size=cf.embedding_dim,
                filter_sizes=list(map(int, cf.filter_sizes.split(','))),
                num_filter=cf.num_filters,
                embeddings=embeddings,
                sentiment_embed_size=cf.sentiment_embedding_length,
                pattern_embed_size=cf.regex_pattern_embedding_length,
                l2_reg_lambda=cf.l2_reg_lambda)

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, sents_batch, patt_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.labels: y_batch,
                    cnn.sentiments: sents_batch,
                    cnn.patterns: patt_batch,
                    cnn.dropout_keep_prob: cf.dropout_keep_prob
                }
                _, step, loss, predictions = sess.run(
                    [cnn.train_op, cnn.global_step, cnn.loss, cnn.predictions],
                    feed_dict=feed_dict
                )
                time_str = datetime.datetime.now().isoformat()

                # print("{}: step{}, loss {:g}".format(time_str, step, loss))
                # print(classification_report(y_batch, predictions))

                return predictions, loss

            def dev_step(x_batch, y_batch, sents_batch, patt_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.labels: y_batch,
                    cnn.sentiments: sents_batch,
                    cnn.patterns: patt_batch,
                    cnn.dropout_keep_prob: 1.0}
                step, loss, predictions = sess.run(
                    [cnn.global_step, cnn.loss, cnn.predictions],
                    feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()

                print("dev: {}: step{}, loss {:g}".format(time_str, step, loss))
                print(classification_report(y_batch, predictions))

                return predictions, loss

            px, py, dx, dy = [], [], [], []
            for ep in range(cf.num_epochs):
                batches = data_helper.Itertool(
                    list(zip(x_train, y_train, sents_train, patt_train)), cf.batch_size, True)
                y_predict_list = []
                y_true_list = []
                x_list = []
                for batch in batches:
                    x_batch, y_batch, sents_batch, patt_batch = zip(
                        *batch)
                    y_true_list.extend(y_batch)
                    predict, loss = train_step(
                        x_batch, y_batch, sents_batch, patt_batch)
                    current_step = tf.train.global_step(sess, cnn.global_step)

                    y_predict_list.extend(predict)
                    x_list.extend(x_batch)
                    py.append(loss)
                    px.append(current_step)
                    if current_step % cf.evaluate_every == 0:
                        print("\nEvaluation:")
                        predictions, loss = dev_step(
                            x_dev, y_dev, sents_dev, patt_dev)
                        print_result(x_dev, predictions, y_dev,
                                     ivocab, label_ivocab, idx)
                        dx.append(current_step)
                        dy.append(loss)
                    if current_step % cf.checkpoint_every == 0:
                        builder = tf.saved_model.builder.SavedModelBuilder(
                            cf.model_path + str(idx) + str(current_step))
                        builder.add_meta_graph_and_variables(
                            sess, ['training_model'])
                        builder.save()
                        print("Saved model after {} time step.".format(current_step))

                print("Train Epoch {} ended!".format(ep))
                # 输出各个类别的prec,rec,f1
                eval = classification_report(y_true_list, y_predict_list)
                print(eval)
        # 画loss的曲线图
        # plt.plot(px, py, label='train')
        # plt.plot(dx, dy, label='dev')
        # plt.legend()
        # plt.show()
        # plt.close('all')
        # 训练完毕,保存模型
        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(cf.model_path, 'model_fold_{}'.format(idx)))
        builder.add_meta_graph_and_variables(
            sess, ['training_model'])
        builder.save()


if __name__ == '__main__':
    #
    train_data_file = sys.argv[1]

    # load data
    print("Loading data...")
    # TODO: 文件位置从外面读取
    filename = os.path.join(os.path.join(cf.data_path, train_data_file))

    X, Y, sentiments, patterns = data_helper.map_file_to_ids(
        filename=filename)
    vocab, ivocab, label_vocab, label_ivocab = data_helper.load_vocab(filename)
    embeddings = data_helper.load_embedding(
        cf.word2vec_path, vocab)
    num_classes = len(label_vocab)
    vocab_size = len(vocab)
    x_fold = cf.num_fold

    # 不进行交叉验证
    if x_fold == 0:
        data_size = len(Y)
        size_per_fold = int(data_size / 10)
        dev_start = 0 * size_per_fold
        dev_end = (0 + 1) * size_per_fold
        x_train, y_train = X[dev_end:], Y[dev_end:]
        sents_train = sentiments[dev_end:]
        patt_train = patterns[dev_end:]
        # dev set
        x_dev = X[dev_start:dev_end]
        y_dev = Y[dev_start:dev_end]
        sents_dev = sentiments[dev_start:dev_end]
        patt_dev = patterns[dev_start:dev_end]
        # 开始训练
        print("{} start {}".format("*" * 10, "*" * 10))
        train(x_train, y_train, sents_train,
              patt_train, x_dev, y_dev, sents_dev, patt_dev, 0)
        print("{} end {}".format("*" * 10, "*" * 10))
        exit()

    data_size = len(Y)
    size_per_fold = int(data_size / x_fold)

    # 交叉验证
    for i in range(x_fold):
        dev_start = i * size_per_fold
        dev_end = (i + 1) * size_per_fold
        # train set
        if i == 0:
            x_train, y_train = X[dev_end:], Y[dev_end:]
            sents_train = sentiments[dev_end:]
            patt_train = patterns[dev_end:]
        elif i == x_fold - 1:
            x_train, y_train = X[:dev_start], Y[:dev_start]
            sents_train = sentiments[:dev_start]
            patt_train = patterns[:dev_start]
        else:
            x_train, y_train = np.concatenate((X[:dev_start], X[dev_end:])), \
                np.concatenate((Y[:dev_start], Y[dev_end:]))
            sents_train = np.concatenate(
                (sentiments[:dev_start], sentiments[dev_end:]))
            patt_train = np.concatenate(
                (patterns[:dev_start], patterns[dev_end:]))
        # dev set
        x_dev = X[dev_start:dev_end]
        y_dev = Y[dev_start:dev_end]
        sents_dev = sentiments[dev_start:dev_end]
        patt_dev = patterns[dev_start:dev_end]
        # 开始训练
        print("{} Turn{} start {}".format("*" * 10, i, "*" * 10))
        train(x_train, y_train, sents_train,
              patt_train, x_dev, y_dev, sents_dev, patt_dev, i)
        print("{} Turn{} end {}".format("*" * 10, i, "*" * 10))