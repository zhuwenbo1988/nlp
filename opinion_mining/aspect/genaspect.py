import os
from collections import defaultdict
import internal_config


PU = internal_config.PU
NN = internal_config.NN
ADJ = internal_config.ADJ
FILTER_THRESHOLD = internal_config.aspect_filter_threshold
THRESHOLD = internal_config.aspect_threshold
WINDOW_SIZE = internal_config.window_size


class AspectExtractor:
    def __init__(
            self,
            resources_data,
            ):
        self.seg_list = resources_data.seg_list
        self.seg_pu_list = resources_data.seg_pu_list
        self.pos_list = resources_data.pos_list
        self.pos_pu_list = resources_data.pos_pu_list
        self.general_opinion = resources_data.general_opinion
        self.stopwords = resources_data.stopwords


    def extract_aspect_by_opinion_window(self):
        """
        开窗口，根据情感词典抽名词得到aspect
        :return:
        """
        print("开始从原始评论中抽取aspect，方法：extract_aspect_by_opinion_window...")
        aspect_dict = defaultdict(lambda: 0)
        for x, WORDs in enumerate(self.seg_list):
            for y, word in enumerate(WORDs):
                if word in self.general_opinion:
                    pre_PU = y #最近的上一个标点符号
                    while self.pos_list[x][pre_PU] != PU and pre_PU > 0:
                        pre_PU -= 1
                    startpoint = y - WINDOW_SIZE if pre_PU not in range(y - WINDOW_SIZE, y) else pre_PU
                    startpoint = startpoint if startpoint >= 0 else 0
                    for i in range(startpoint, y):
                        if self.pos_list[x][i] == NN:
                            aspect_dict[self.seg_list[x][i]] += 1

        temp = list(filter(lambda x: x[0] not in self.stopwords, aspect_dict.items()))  # 删去停用词
        temp = list(filter(lambda x: len(x[0]) > 1, temp))  # 删去一个字的词
        temp = list(filter(lambda x: x[1] > FILTER_THRESHOLD, temp))
        aspect_list = [item[0] for item in temp]
        print("---抽取aspect结束，共抽取到%s个aspect---" % (len(aspect_list)))
        return aspect_list
