"""内部过程参数配置"""

"""分词，词性标注，依存分析接口地址"""
SEG_POS_DEP_URL = "http://wordseg-postag-depparser-ws/nlp_outfit?query="
PAIR_POLARITY_URL = "http://mobvoi-oss/v1/ufile/mobvoi-nlp-private/opinion_mining/pair_polarity_dict"
WORD2VEC_URL = "http://mobvoi-oss/v1/ufile/mobvoi-nlp-private/opinion_mining/word2vec.model"

"""词性符号的定义"""
PU = "PU"
NN = "NN"
ADJ = ["VA", "JJ"]

"""保留的标点符号"""
keep_p = ['，', '。', '！', '？']
PU_pattern = "，|。|！|？"

"""窗口大小设置"""
window_size = 5  # <- 抽取aspect，opinion过程中窗口的大小
clue_max_size = 10  # <-抽取短语时，短语的最大长度

"""aspect抽取"""
aspect_filter_threshold = 9  # <- 用通用情感词典抽取aspect过程中过滤的阈值"""
aspect_threshold = 100  # <- 抽取aspect时只取前100个

"""pair build阶段配置"""
negation = ["不", "不是", "不太", "没", "没有", "无"]
aspect_do_not_use = ["是", "说", "时候", "免费", "美", "会", "缺点"]
opinion_do_not_use = ["最", "不", "很"]
pattern_do_not_use = ["的-", "和-", "和+", "而+", "而-", "又+", "又-", "而且+", "而且-"]

"""pair refine阶段配置"""
split_point = 0.1  # <- 高置信度pair阈值
pair_refine_point = 0.3  # <- pair refine置信度阈值

"""未知极性"""
polarity_unk = '2'

