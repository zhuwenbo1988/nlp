# /bin/python2.7

import sys
import os
from preparation import *
from preprocess import *

if __name__ == '__main__':
    basedir = sys.argv[1]

    prepare = Preparation()
    data_path = os.path.join(basedir, "dataset.txt")
    corpus_path = os.path.join(basedir, "corpus.txt")
    word_dict_path = os.path.join(basedir, "word_dict.txt")
    corpus_preprocessed = os.path.join(basedir, "corpus_preprocessed.txt")

    corpus, rels = prepare.run_with_one_corpus(data_path)
    print('total corpus : %d ...' % (len(corpus)))
    print('total relations : %d ...' % (len(rels)))
    prepare.save_corpus(corpus_path, corpus)

    rel_train, rel_valid, rel_test = prepare.split_train_valid_test_for_ranking(rels,
                                                            [1000000, 50001, 6660])
    prepare.save_relation(os.path.join(basedir, "relation.train.fold1.txt"), rel_train)
    prepare.save_relation(os.path.join(basedir, "relation.valid.fold1.txt"), rel_valid)
    prepare.save_relation(os.path.join(basedir, "relation.test.fold1.txt"), rel_test)
    print('preparation finished ...')

    # Prerpocess corpus file
    preprocessor = Preprocess()
    dids, docs, word_dict = preprocessor.run(corpus_path)
    preprocessor.save_word_dict(word_dict_path)

    preprocessor.save_id2doc(corpus_preprocessed, dids, docs)
    print('preprocess finished ...')