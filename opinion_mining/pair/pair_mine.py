import internal_config


SPLIT_POINT = internal_config.split_point


class PairMine:
    def __init__(self, pair_sorted_socre, word2vec_model):
        self.word2vec_model = word2vec_model
        self.pair_sorted_score = pair_sorted_socre
        self.base_line = SPLIT_POINT * len(self.pair_sorted_score)

    def aspect_mine(self):
        print("开始进行pair抽取阶段三:aspect提炼...")
        pair_mine_score = {}
        pair_mine_count = {}
        pair_useful = {}  # 论文中前10%，置信度较高的集和
        pair_useless = {}  # 论文中前10%之后的，置信度不高的集和
        #{'n\ts': 300, 'n\ts': 200}
        for idx, line in enumerate(self.pair_sorted_score):
            n, s = line[0].split('\t')
            score = line[1]
            if idx >= self.base_line:
                if s not in pair_useless:
                    pair_useless[s] = {}
                    pair_useless[s][n] = score
                elif n not in pair_useless[s]:
                    pair_useless[s][n] = score
            elif idx < self.base_line:
                if s not in pair_useful:
                    pair_useful[s] = {}
                    pair_useful[s][n] = score
                elif n not in pair_useful[s]:
                    pair_useful[s][n] = score
        for s in pair_useless:
            n_renew = {}
            if s in pair_useful:
                for word in pair_useful[s]:
                    n_renew[word] = 0.0
            for n in pair_useless[s]:
                self._get_score(n, n_renew, pair_mine_score, pair_mine_count, n, s)

        pair_mine_sort = sorted(pair_mine_score.items(), key=lambda d: d[1], reverse=True)
        print("---pair抽取阶段三完成---")
        return self._result(pair_mine_sort, pair_mine_count)

    def opinion_mine(self):
        print("开始进行pair抽取阶段四:opinion提炼...")
        pair_mine_score = {}
        pair_mine_count = {}
        pair_useful = {}  # 论文中前10%，置信度较高的集和
        pair_useless = {}  # 论文中前10%之后的，置信度不高的集和

        for idx, line in enumerate(self.pair_sorted_score):
            n, s = line[0].split('\t')
            score = line[1]
            if idx >= self.base_line:
                if n not in pair_useless:
                    pair_useless[n] = {}
                    pair_useless[n][s] = score
                elif s not in pair_useless[n]:
                    pair_useless[n][s] = score
            elif idx < self.base_line:
                if n not in pair_useful:
                    pair_useful[n] = {}
                    pair_useful[n][s] = score
                elif s not in pair_useful[n]:
                    pair_useful[n][s] = score
        """这些opinion都是用来修饰A的，如果新的opinion和pair_useful中的修饰A的opinion相似度超过某个阈值那么就把这个pair加入候选"""
        for n in pair_useless:
            s_renew = {}
            if n in pair_useful:
                for word in pair_useful[n]:
                    s_renew[word] = 0.0  # 这个s_renew中存的是这个在pair_useless中的aspect如果也在pair_useful中，那么在pair_useful中全部的opinon
            for s in pair_useless[n]:  # 对n中的每个s
                self._get_score(s, s_renew, pair_mine_score, pair_mine_count, n, s)
        pair_mine_sort = sorted(pair_mine_score.items(), key=lambda d: d[1], reverse=True)
        print("---pair抽取阶段四完成---")
        return self._result(pair_mine_sort, pair_mine_count)

    def _get_score(self, target, renew, pair_mine_score, pair_mine_count, n, s):
        for sim in renew:
            if target not in self.word2vec_model \
                    or sim not in self.word2vec_model:
                continue
            renew[sim] = self.word2vec_model.wv.similarity(target, sim)
        sum = 0
        for sim in renew:
            sum += renew[sim]
        numb = len(renew)
        if numb:
            score = sum / len(renew)
        else:
            score = 0.0
        pair_mine_score[n + '\t' + s] = score
        pair_mine_count[n + '\t' + s] = numb

    def _result(self, pair_mine_sort, pair_mine_count):
        res_list = []
        for item in pair_mine_sort:
            pair = item[0]
            score = item[1]
            count = pair_mine_count[item[0]]
            res_list.append("{}\t{}\t{}".format(pair, score, count))
        return res_list



