# -*- coding: utf-8 -*-

import codecs
import hashlib


class Preparation(object):

    def get_text_id(self, hashid, text, idtag='T'):
        hash_obj = hashlib.sha1(text.encode('utf8'))  # if the text are the same, then the hash_code are also the same
        hex_dig = hash_obj.hexdigest()
        if hex_dig in hashid:
            return hashid[hex_dig]
        else:
            tid = idtag + str(len(hashid))  # start from 0, 1, 2, ...
            hashid[hex_dig] = tid
            return tid

    def parse_line(self, line, delimiter='\t'):
        subs = line.split(delimiter)
        if 3 != len(subs):
            print(subs[1])
            raise ValueError('format of data file wrong, should be \'label,text1,text2\'.')
        else:
            return subs[0], subs[1], subs[2]

    def run_with_one_corpus(self, file_path):
        hashid = {}
        corpus = {}
        rels = []
        f = codecs.open(file_path, 'r', encoding='utf8')
        for line in f:
            line = line.strip()
            label, t1, t2 = self.parse_line(line)
            id1 = self.get_text_id(hashid, t1, 'T')
            id2 = self.get_text_id(hashid, t2, 'T')
            corpus[id1] = t1
            corpus[id2] = t2
            rels.append((label, id1, id2))
        f.close()
        return corpus, rels

    @staticmethod
    def save_corpus(file_path, corpus):
        f = codecs.open(file_path, 'w', encoding='utf8')
        for qid, text in corpus.items():
            f.write('%s %s\n' % (qid, text))
        f.close()

    @staticmethod
    def save_relation(file_path, relations):
        f = open(file_path, 'w')
        for rel in relations:
            f.write('%s %s %s\n' % (rel))
        f.close()

    @staticmethod
    def split_train_valid_test_for_ranking(relations, ratio=(6, 3, 3)):
        qid_group = []
        # random.shuffle(qid_group)
        total_rel = len(relations)
        for i in range(total_rel):
            qid_group.append(i)
        num_train = int(ratio[0])
        num_valid = int(ratio[1])
        valid_end = num_train + num_valid
        qid_train = qid_group[: num_train]
        qid_valid = qid_group[num_train: valid_end]
        qid_test = qid_group[valid_end:]
        def select_rel_by_qids(qids):
            rels = []
            qids = set(qids)
            count = 0
            for r, q, d in relations:
                if count in qids:
                    rels.append((r, q, d))
                count += 1
            return rels
        rel_train = select_rel_by_qids(qid_train)
        rel_valid = select_rel_by_qids(qid_valid)
        rel_test = select_rel_by_qids(qid_test)
        return rel_train, rel_valid, rel_test