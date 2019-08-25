# -*- coding:utf-8 -*-
# @Copyright 2019 mobvoi Inc.
# @Author   :wbzhu


import sys


class FilterCN():

    def __init__(self, sw_file, max_langth=sys.maxint):
        l = []
        for line in open(sw_file):
            word = line.strip().decode('utf-8')
            l.append(word)
        self.stop_words = set(l)
        self.max_langth = max_langth

    def filter(self, words):
        result = []
        for word in words:
            if not word:
                continue
            if word == ' ':
                continue
            if word in self.stop_words:
                continue
            if word == 'NP':
                continue
            if word == 'N':
                continue
            if word == 'E':
                continue
            if word == 'B':
                continue
            if word == 'D':
                continue
            result.append(word)
        if len(result) > self.max_langth:
            return []
        return result