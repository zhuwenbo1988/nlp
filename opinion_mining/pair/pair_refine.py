import internal_config
import requests


SPLIT_POINT = internal_config.split_point
POLARITY_UNK = internal_config.polarity_unk
PAIR_POLARITY_URL = internal_config.PAIR_POLARITY_URL

class PairRefine:
    def __init__(self, general_opinion, pair_sorted_ns, pair_for_mine_aspect, pair_for_mine_opinion):
        self.general_opinion = general_opinion
        self.pair_sorted_ns = pair_sorted_ns
        self.pair_for_mine_aspect = pair_for_mine_aspect
        self.pair_for_mine_opinion = pair_for_mine_opinion

    """过滤函数，卡的阈值为0.3"""
    def _pair_filter(self, for_refine, refine_point):
        refine_result = []
        for idx, line in enumerate(for_refine):
            p_l = line.split('\t')
            if float(p_l[2]) < refine_point:  # <- 这里p_l[2]就是排序算法得到的pair的分数
                break
            else:
                refine_result.append(line)
        return refine_result

    def _build_pair_dict(self, f):
        dc = {}
        for line in f:
            n, s, score, c = line.split('\t') # aspect，opinion，分数（出现次数/置信度高的某个aspect或者opinion的全部opinion或者aspect），出现次数
            dc[n + '\t' + s] = float(score)
        return dc

    def refine(self):
        pair_for_mine_aspect_result = self._pair_filter(self.pair_for_mine_aspect, 0.1)
        pair_for_mine_opinion_result = self._pair_filter(self.pair_for_mine_opinion, 0.3)

        refine_aspect = self._build_pair_dict(pair_for_mine_aspect_result)  # 提炼出的aspect得到的pair
        refine_opinion = self._build_pair_dict(pair_for_mine_opinion_result)  # 提炼出的opinion得到的pair
        """去两个结果的交集"""
        refine_pair = []
        for pair in refine_aspect:
            if pair in refine_opinion:
                refine_pair.append("{}\t{}\t{}".format(pair, str(refine_aspect[pair]), str(refine_opinion[pair])))
        return refine_pair

    def gen_pair_polarity(self, refine_pair):
        """
        得到build阶段最终输出的pair对，并给已知极性的标上极性
        :return: 
        """
        final_pair = []
        base_line = SPLIT_POINT * len(self.pair_sorted_ns)
        for idx, line in enumerate(self.pair_sorted_ns):
            n, s = line[0].split('\t')
            score = line[1]
            if idx <= base_line:
                final_pair.append([n, s])
        for idx, line in enumerate(refine_pair):
            n, s, _, _ = line.split('\t')
            final_pair.append([n, s])
        print("---pair抽取模块结束，共抽取到%s个表达---" % len(final_pair))
        pair_polarity_list = []
        for item in final_pair:
            if item[1] in self.general_opinion:
                pair_polarity_list.append("{}\t{}".format(str(item), str(self.general_opinion[item[1]])))
            else:
                pair_polarity_list.append("{}\t{}".format(str(item), POLARITY_UNK))
        return pair_polarity_list

    def gen_final_build_result_and_save_in_db(self, domain, collection, pair_polarity_list):
        """写到数据库"""
        pair_polarity = {}
        for idx, line in enumerate(pair_polarity_list):
            pair, polarity = line.split("\t")
            pair = eval(pair)
            if pair[0] not in pair_polarity:
                pair_polarity[pair[0]] = {}
            pair_polarity[pair[0]][pair[1]] = polarity

        query = {
            'domain': domain,
            'pair_polarity': pair_polarity
        }

        res_dict = {'same_aspects': {}, 'add_aspects': {}, 'delete_aspects': {}}
        """获取旧版的pair并与新版的pair进行diff"""
        old_pair_polarity = collection.find_one({'domain': domain})['pair_polarity']
        old_aspects = set(old_pair_polarity.keys())
        new_aspects = set(pair_polarity.keys())
        same_aspects = old_aspects & new_aspects
        dif_aspects = old_aspects ^ new_aspects
        for same_aspect in same_aspects:
            same_opinions, add_opinions, delete_opinions = self.differ_dict(old_pair_polarity[same_aspect], pair_polarity[same_aspect])
            res_dict['same_aspects'][same_aspect] = {'same_opinions': same_opinions,
                                                     'add_opinions': add_opinions,
                                                     'delete_opinions': delete_opinions}
        for aspect in dif_aspects:
            if aspect in pair_polarity:
                res_dict['add_aspects'][aspect] = [opinion for opinion in pair_polarity[aspect]]
            else:
                res_dict['delete_aspects'][aspect] = [opinion for opinion in old_pair_polarity[aspect]]


        """删去旧版的pair"""
        collection.delete_one({'domain': domain})
        collection.insert_one(query)

        return res_dict, len(pair_polarity_list)

    def differ_dict(self, dict1, dict2):
        #dict1-旧版，dict2-新版
        if isinstance(dict1, dict):
            dict1_keys = set(dict1.keys())
            dict2_keys = set(dict2.keys())
            same_keys = list(dict1_keys & dict2_keys)
            dif_keys = dict1_keys ^ dict2_keys
            add_keys = []
            delete_keys = []
            for key in dif_keys:
                if key in dict2:
                    add_keys.append(key)
                else:
                    delete_keys.append(key)
            return same_keys, add_keys, delete_keys


    @classmethod
    def merge_with_annotation(clf, mongodb_client, domain):
        """将标注结果与数据库中的pair_polarity进行合并"""
        """加载人工标注的情感词表"""
        r = requests.get(PAIR_POLARITY_URL)
        biaozhu_pair_polarity = {}
        for line in r.content.decode('utf-8').split('\n'):
            if line == "":
                continue
            item = eval(line)
            aspect = item[0]
            opinion_dict = item[1]
            if aspect not in biaozhu_pair_polarity:
                biaozhu_pair_polarity[aspect] = opinion_dict

        build_pairs_query_res = mongodb_client.db['opinion_build_pairs'].find_one({'domain': domain})
        new_pair_polarity = build_pairs_query_res['pair_polarity']

        for aspect in biaozhu_pair_polarity:
            if aspect in new_pair_polarity:
                for opinion in biaozhu_pair_polarity[aspect]:
                    if opinion in new_pair_polarity[aspect]:
                        new_pair_polarity[aspect][opinion] = biaozhu_pair_polarity[aspect][opinion]

        query = {
            'domain': domain,
            'pair_polarity': new_pair_polarity
        }
        """删去旧版的pair"""
        mongodb_client.db['opinion_build_pairs'].delete_one({'domain': domain})
        res_str = mongodb_client.db['opinion_build_pairs'].insert_one(query)
        return res_str


