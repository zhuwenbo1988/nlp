# encoding:utf-8


import os


class GlobalConfig(object):
    data_path = './data/'
    model_path = './models/'

    word2vec_path = "./word2vec/word_vectors_300.txt"

    all_vocab_path = './vocab/'

    regex_path = os.path.join(all_vocab_path, 'regex_pattern_v1')

    vocab_sentiment_happy = os.path.join(all_vocab_path, 'happy')
    vocab_sentiment_angry = os.path.join(all_vocab_path, 'angry')
    vocab_sentiment_sad = os.path.join(all_vocab_path, 'sad')
    vocab_sentiment_fear = os.path.join(all_vocab_path, 'fear')
    vocab_sentiment_surprise = os.path.join(all_vocab_path, 'surprise')

    vocab_path = "./data/vocab"
    label_path = "./data/label"

    sentiment_lexicon = './data/sentiment_dict.xlsx'

    # input length
    max_sequence_length = 30
    max_sentiment_len = 6
    max_pattern_length = 6

    # embedding length
    embedding_dim = 300
    sentiment_embedding_length = 100
    regex_pattern_embedding_length = 100

    # tf session config params
    allow_soft_placement = True
    log_device_placement = False

    # NN params
    filter_sizes = '3,4,5'
    num_filters = 64
    dropout_keep_prob = 0.5

    # train params
    batch_size = 32
    num_epochs = 1
    num_fold = 0
    evaluate_every = 10
    checkpoint_every = 1000

    # other params
    l2_reg_lambda = 0.0
