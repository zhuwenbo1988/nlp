import internal_config
import os
import random

class PairPattSort():
    '''
    Pair-Patt-Count structure
    '''
    def __init__(self, ns_dict):
        self._get_map(ns_dict)

    def _get_map(self, ns_dict):
        '''
        get map: [pair-patt], [patt-pair], [pair](score), [patt](score)

        :param ns_dict: Entity.str { Emotion.str { Pattern.str { Count.int (It's a three-level hash structure)
        :return:
        '''
        pair_list = []
        patt_dict = {}
        patt_pair_map = {}
        pair_patt_map = {}

        aspects = list(ns_dict.keys())
        random.shuffle(aspects)

        for n in aspects:
            for s in ns_dict[n]:
                n_s = "{}\t{}".format(n, s)   #这里存的pair是字符串，中间用\t隔开
                pair_list.append(n_s)
                pair_patt_map[n_s] = {}
                for patt in ns_dict[n][s]:
                    if patt not in patt_dict:
                        patt_dict[patt] = 1.0
                    pair_patt_map[n_s][patt] = ns_dict[n][s][patt]
                    if patt in patt_pair_map:
                        patt_pair_map[patt][n_s] = ns_dict[n][s][patt]
                    else:
                        patt_pair_map[patt] = {}
                        patt_pair_map[patt][n_s] = ns_dict[n][s][patt]
        self.patt_pair_map = patt_pair_map
        self.pair_patt_map = pair_patt_map
        self.pair_len = len(pair_list)
        self.patt_len = len(patt_dict)
        self.pair_score = dict([(word, 1.) for i, word in enumerate(pair_list)])
        self.patt_score = patt_dict

    """"正则化，和为score_len"""
    def _norm(self, score_dict, score_len):
        sum_score = 0.
        for s in score_dict:
            sum_score += score_dict[s]
        for s in score_dict:
            score_dict[s] = score_dict[s] / sum_score * score_len
        return score_dict


    def _patt_pair(self):
        for pair in self.pair_patt_map:  # <- 循环遍历每个pair
            value = 0.
            for patt in self.pair_patt_map[pair]:  # <- 每个pair中的pattern出现的个数 * 这个pattern的score，然后求和得到这个pair的分数
                value += self.pair_patt_map[pair][patt] * self.patt_score[patt]
            self.pair_score[pair] = value

    def _pair_patt(self):
        for patt in self.patt_pair_map:  # <- 遍历每个pattern
            value = 0.
            for pair in self.patt_pair_map[patt]:  # <- 每个被pattern修饰的pair出现的个数 * 这个pair的score，然后求和得到这个pattern1的
                value += self.patt_pair_map[patt][pair] * self.pair_score[pair]
            self.patt_score[patt] = value

    def _patt_correct(self):
        self.patt_score['的-'] = 0.0

    def _iterative(self):
        '''
        A complete iteration
        [pair] = [patt-pair] * [patt]
        [patt] = [pair-patt] * [pair]
        :return:
        '''
        self._patt_pair()
        self.pair_score = self._norm(self.pair_score, self.pair_len)
        self._pair_patt()
        self.patt_score = self._norm(self.patt_score, self.patt_len)

    def pair_sort(self):
        print("开始进行pair抽取阶段二：组合排序...")
        for i in range(100):
            self._iterative()
        pair_score = sorted(self.pair_score.items(), key=lambda d: d[1], reverse=True)
        print("---pair抽取阶段二完成---")
        return pair_score
