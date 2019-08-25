from collections import defaultdict
import math
from pprint import pprint


class Summary:
    def __init__(self,comment_docs):
        self.comment_docs = comment_docs

    @classmethod
    def get_statistical_analysis(clf, comments_docs):
        tag_statis_dict = {}
        for idx, doc in enumerate(comments_docs):
            try:
                pairs = doc['pairs']
                for pair in pairs:
                    tag = pair['tag']
                    polarity = int(pair['polarity'])
                    if tag not in tag_statis_dict:
                        tag_statis_dict[tag] = {}
                        tag_statis_dict[tag]['pos'] = 0
                        tag_statis_dict[tag]['neg'] = 0
                    if polarity > 0:
                        tag_statis_dict[tag]['pos'] += 1
                    elif polarity < 0:
                        tag_statis_dict[tag]['neg'] += 1
            except:
                pass
        # 百分比格式
        # for tag in tag_statis_dict:
        #     p = tag_statis_dict[tag]['pos']
        #     n = tag_statis_dict[tag]['neg']
        #     s = p + n
        #     tag_statis_dict[tag]['pos'] = round(p / s * 100, 2)
        #     tag_statis_dict[tag]['neg'] = round(n / s * 100, 2)
        pprint(tag_statis_dict)
        return tag_statis_dict




