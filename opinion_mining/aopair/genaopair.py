import configparser
import internal_config
import re
from pymongo import UpdateOne
from collections import defaultdict
import numpy as np

PU = internal_config.PU
NEGATION = ["不", "不是", "不太", "没", "没有", "无"]
KEEP_P = internal_config.keep_p
WINDOW_SIZE = internal_config.window_size
CLUE_MAX_SIZE = internal_config.clue_max_size


class AOPair:
    def __init__(
            self,
            comment_docs,
            pair_polarity,
            resources,
            ):
        self.comment_docs = comment_docs
        self.pair_polarity = pair_polarity
        self.word2vec_model = resources.word2vec_model
        self.user_defined_aspect = resources.user_defined_aspect

    def pair_tag_classify(self, aspect_l):
        """利用词向量计算短语标签属于哪个用户定义的tag"""
        raw_aspect_phrase_vector = sum(np.asarray([self.word2vec_model[word] for word in aspect_l])) / len(aspect_l)

        user_defined_aspect_vector_dict = {}
        for word in self.user_defined_aspect:
            user_defined_aspect_vector_dict[word] = self.word2vec_model.wv[word]

        sim = defaultdict(lambda: 0)
        for n in user_defined_aspect_vector_dict:
            vector1 = user_defined_aspect_vector_dict[n]
            vector2 = raw_aspect_phrase_vector
            sim[n] = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        most_sim_udf = max(sim, key=sim.get)
        return most_sim_udf

    def inference_res_save_in_db(self, collection):
        print("开始从原始评论中进行观点挖掘...")
        size = len(self.comment_docs)
        query_requests = []
        for line_idx, doc in enumerate(self.comment_docs):
            """每条评论按照标点切分成每个短句处理"""
            line = doc['seg'].split(' ')
            cur_doc_res = self.extract_pairs(line)
            if cur_doc_res:
                query_requests.append(
                    UpdateOne(
                        {'_id': doc['_id']},
                        {'$set': {'pairs': cur_doc_res}},
                        upsert=True))
        res = collection.bulk_write(query_requests)
        res_str = '数据库更新结果:{},观点挖掘模块结束，共覆盖{}%的评论' \
            .format(res.bulk_api_result, round(res.modified_count / size * 100, 2))
        return res_str

    def get_infer_one_res(self):
        try:
            assert type(self.comment_docs) is str
        except Exception as e:
            return 'error:{}'.format(e)
        print('self.comment_docs:{}'.format(self.comment_docs))
        cur_doc_res = self.extract_pairs(self.comment_docs.split(' '))
        return cur_doc_res

    def extract_pairs(self, line):
        cur_line = []
        last_end = 0
        for idx, w in enumerate(line):
            if w in KEEP_P:
                cur_line.append(line[last_end:idx])
                last_end = idx + 1
            elif idx == len(line) - 1:
                cur_line.append(line[last_end:])

        cur_doc_res = []
        for clue in cur_line:
            """假设：两个逗号之间的短语就是一个观点表达，一般人不会在一个观点表达中说同一个形容词两次"""
            opinion_used = defaultdict(list)
            result_to_write = {}

            for word_index, word in enumerate(clue):
                if word in self.pair_polarity:
                    """向前开窗口"""
                    startpoint = word_index - WINDOW_SIZE if word_index - WINDOW_SIZE > 0 else 0
                    for i in range(startpoint, word_index):
                        opinion = clue[i]
                        if opinion in self.pair_polarity[word] and clue[i + 1] == "的":  # 形容词之后紧跟"的"是正常的表达
                            raw_express = [i, word_index + 1]
                            opinion_used[opinion].append(word_index)
                            """判断极性"""
                            polarity = int(self.pair_polarity[word][opinion])
                            result_to_write[word_index] = [polarity, raw_express]

                    """向后开窗口"""
                    endpoint = word_index + CLUE_MAX_SIZE if word_index + CLUE_MAX_SIZE < len(clue) else len(clue)
                    for i in range(word_index + 1, endpoint):
                        opinion = clue[i]
                        if opinion in self.pair_polarity[word]:
                            """记录原始表达的开始位置和结束位置"""
                            raw_express = [word_index, i + 1]

                            """判断极性以及组合否定表达"""
                            polarity = int(self.pair_polarity[word][opinion])
                            express_clue = clue[word_index: i + 1]  # [aspect -> opinion]
                            for j, _w in enumerate(express_clue):
                                if _w in NEGATION:
                                    opinion = "".join(express_clue[j:])
                                    polarity = -polarity

                            """记录被使用过的opinion"""
                            opinion_used[opinion].append(word_index)
                            result_to_write[word_index] = [polarity, raw_express]

            """对用户描述的相同对象进行组合,并输出最终结果"""
            for opinion in opinion_used:
                first_aspect_idx = opinion_used[opinion][0]
                last_aspect_idx = opinion_used[opinion][-1]

                aspect_l = [clue[a] for a in opinion_used[opinion]]

                raw_express = "".join(clue[result_to_write[first_aspect_idx][1][0]: result_to_write[last_aspect_idx][1][1]])
                polarity = result_to_write[first_aspect_idx][0]

                tag = self.pair_tag_classify(aspect_l)

                cur_doc_res.append({'aspect': aspect_l,
                                    'opinion': opinion,
                                    'polarity': polarity,
                                    'raw_express': raw_express,
                                    'tag': tag})
        return cur_doc_res
