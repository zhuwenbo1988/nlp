import requests
import json
import re
import internal_config
from pymongo import UpdateOne, DeleteOne
from logger import logger

SEG_POS_DEP_URL = internal_config.SEG_POS_DEP_URL


FH_PUNCTUATION = [
    (u"。", u"."), (u"，", u","), (u"！", u"!"), (u"？", u"?"), (u"”", u'"'),
    (u"'", u"'"), (u"‘", u"`"), (u"＠", u"@"), (u"＿", u"_"), (u"：", u":"),
    (u"；", u";"), (u"＃", u"#"), (u"＄", u"$"), (u"％", u"%"), (u"＆", u"&"),
    (u"（", u"("), (u"）", u")"), (u"‐", u"-"), (u"＝", u"="), (u"＊", u"*"),
    (u"＋", u"+"), (u"－", u"-"), (u"／", u"/"), (u"＜", u"<"), (u"＞", u">"),
    (u"［", u"["), (u"￥", u"\\"), (u"］", u"]"), (u"＾", u"^"), (u"｛", u"{"),
    (u"｜", u"|"), (u"｝", u"}"), (u"～", u"~"),
]

f2h = {}
for item in FH_PUNCTUATION:
    c1 = item[0]
    c2 = item[1]
    f2h[c2] = c1


def convert(content):
    nc = []
    for c in content:
        if c in f2h:
            nc.append(f2h[c])
            continue
        nc.append(c)
    return "".join(nc)

keep_p = internal_config.keep_p #保留的标点符号
PU_pattern = internal_config.PU_pattern #用于分割句子的标点符号


def clean(line):
    if line == "":
        return
    line = convert(line)
    c_content = []
    for char in line:
        if re.search("[\u4e00-\u9fa5]", char):
            c_content.append(char)
        elif re.search("[\w]", char):
            continue
        elif char in keep_p:
            c_content.append(char)
        else:
            c_content.append(' ')
    nc_content = []
    c = 0
    for char in c_content:
        if char == ' ' or char in keep_p:
            c += 1
        else:
            c = 0
        if c < 2:
            nc_content.append(char)
    result = ''.join(nc_content)
    result = result.strip()
    result = result.replace(' ', '，')
    return result


def get_pos_seg_praser(text):
    """
    获取分词，磁性标注，依存分析结果
    :param text:
    :return:
    """
    param = text
    all_url = SEG_POS_DEP_URL + param
    try:
        req = requests.get(all_url)
        json_respones = req.content.decode()
        dict_json = json.loads(json_respones)
        return dict_json["word_segments"], dict_json["pos_tag"]
    except Exception as e:
        print(e)
        return 0


def process_raw_data_and_save_in_db(collection, query):
    logger.info("开始预处理原始评论数据，进行分词，词性标注，依存分析...")
    raw_documents = list(collection.find(query))
    requests = []
    docs_for_delete = []
    error_count = 0
    for idx, doc in enumerate(raw_documents):
        comment_str = doc['raw_comment']
        comment_str = clean(comment_str)
        result = get_pos_seg_praser(comment_str)  # <- result[0]:seg, result[1]:pos
        if result == 0:
            docs_for_delete.append(DeleteOne({'_id': doc['_id']}))
            logger.info("噪声评论:{}".format(doc))
            error_count += 1
        else:
            requests.append(
                UpdateOne({'_id': doc['_id']},
                          {'$set': {'clean_comment': comment_str, 'seg': result[0], 'pos': result[1]}},
                          upsert=True))
        if idx % 1000 == 0:
            logger.info("---processed {} docs---".format(idx))
    if docs_for_delete:
        r = collection.bulk_write(docs_for_delete)
        logger.info('删除噪声评论:{}'.format(r.bulk_api_result))
    if requests:
        res = collection.bulk_write(requests)
        res_str = "---原始评论数据预处理完毕，原始评论共{}条, 有{}条无结果, mongoDB log:{}---".format(len(raw_documents), error_count, res.bulk_api_result)
        return res_str
    else:
        return '待处理的新评论数量为0'

