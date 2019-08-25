import internal_config


ADJ = internal_config.ADJ
WINDOW_SIZE = internal_config.window_size
ASPECT_DO_NOT_USE = internal_config.aspect_do_not_use
OPINION_DO_NOT_USE = internal_config.opinion_do_not_use
PATTERN_DO_NOT_USE = internal_config.pattern_do_not_use


class AspectFilter:
    def __init__(self, seg_pu_list, pos_pu_list):
        self.seg_pu_list = seg_pu_list
        self.pos_pu_list = pos_pu_list
        self.ns_dict = {}

    def _seg2nsd(self, aspect_for_filter):
        print("开始进行pair抽取阶段一：抽取pair和pattern...")
        for x, clue in enumerate(self.seg_pu_list):
            N_list = []
            S_list = []
            word_list = clue
            for y, word in enumerate(clue):
                if word in aspect_for_filter:
                    N_list.append(y)
                elif self.pos_pu_list[x][y] in ADJ:
                    S_list.append(y)
            if N_list and S_list:
                self._make_nsdict(word_list, N_list, S_list)

    def _make_nsdict(self, word_list, N_list, S_list):
        for n in N_list:
            for s in S_list:
                if (1 < n - s < WINDOW_SIZE + 1) or (1 < s - n < WINDOW_SIZE + 1):  # 窗口大小是5
                    if word_list[n] not in self.ns_dict:
                        self.ns_dict[word_list[n]] = {}
                    if word_list[s] not in self.ns_dict[word_list[n]]:
                        self.ns_dict[word_list[n]][word_list[s]] = {}
                    if n > s:
                        patt = ' '.join(word_list[s + 1: n]) + '+'
                    else:
                        patt = ' '.join(word_list[n + 1: s]) + '-'
                    if patt not in self.ns_dict[word_list[n]][word_list[s]]:
                        self.ns_dict[word_list[n]][word_list[s]][patt] = 0.
                    self.ns_dict[word_list[n]][word_list[s]][patt] += 1.

    def _noise_del(self):
        for aspect in ASPECT_DO_NOT_USE:
            self._noise(aspect, self.ns_dict)
        for n in self.ns_dict:
            for opinion in OPINION_DO_NOT_USE:
                self._noise(opinion, self.ns_dict[n])
            for s in self.ns_dict[n]:
                for pattern in PATTERN_DO_NOT_USE:
                    self._noise(pattern,self.ns_dict[n][s])

    def _noise(self, str, dict):
        if str in dict:
            del dict[str]

    def build_nsdict(self, aspect_for_filter):
        self._seg2nsd(aspect_for_filter)
        self._noise_del()
        print("---抽取pair阶段一完成---")
        return self.ns_dict
