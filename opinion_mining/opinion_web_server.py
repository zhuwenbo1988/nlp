from datetime import datetime
import time
import pymongo
from preprocess import processor
from aspect import genaspect
from pair import build_pair_flow
from aopair import genaopair
import json
import utils
from web_service.http_server import OMHTTPServer
from database import mongodb_client
from summary import gensummary
import requests
from logger import logger
import sys


"""毫秒级时间区间"""
ONE_WEEK_LENGTH = 1000 * 60 * 60 * 24 * 7
ONE_DAY_LENGTH = 1000 * 60 * 60 * 24

"""预加载资源"""
SPU_LIST = []
SKU_DICT = {}
TAG_LIST = []
"""加载infer one的资源"""
class Resources_for_infer_one:
    __slots__ = ['general_opinion', 'user_defined_aspect', 'word2vec_model', 'pair_polarity']
    def __init__(self):
        self.general_opinion = None
        self.user_defined_aspect = None
        self.word2vec_model = None
        self.pair_polarity = None

res_for_one = Resources_for_infer_one()


def load_spu_list():
    global SPU_LIST
    try:
        query_res = mongodb_client.db['opinion_resources'].find_one({'doc_type': 'spu_list'})
        SPU_LIST = query_res['lexicon']
    except Exception as e:
        logger.info("load_spu_list error:".format(e))

def load_sku_list():
    global SKU_DICT
    try:
        query_res = mongodb_client.db['opinion_resources'].find_one({'doc_type': 'sku_list'})
        SKU_DICT = query_res['lexicon']
    except Exception as e:
        logger.info("load_sku_list error:".format(e))


def load_tag_list(domain):
    global TAG_LIST
    try:
        query_res = mongodb_client.db['opinion_resources'].find_one({'doc_type': 'product_tag', 'domain': domain})
        TAG_LIST = query_res['lexicon']
    except Exception as e:
        logger.info("load_tag_list error:".format(e))


def load_resources_for_infer_one(domain):
    """加载第三方情感词典"""
    general_opinion_doc = mongodb_client.db['opinion_resources'].find_one({'doc_type': 'general_opinion'})
    res_for_one.general_opinion = general_opinion_doc['lexicon']
    """加载用户定义词典"""
    res_for_one.user_defined_aspect = TAG_LIST

    """加载word2vec模型"""
    res_for_one.word2vec_model = utils.load_word2vec_model()

    """加载build完成的pair"""
    build_pairs_query_res = mongodb_client.db['opinion_build_pairs'].find_one({'domain': domain})
    pair_polarity = build_pairs_query_res['pair_polarity']
    res_for_one.pair_polarity = pair_polarity


def timestamp2date(timestr):
    time_array = time.localtime(int(timestr) / 1000)
    time_str = time.strftime("%Y-%m-%d", time_array)
    year, month, day = time_str.split('-')
    return year, month, day


def timestamp2str(timestr):
    time_array = time.localtime(int(timestr) / 1000)
    time_str = time.strftime("%Y-%m-%d", time_array)
    return time_str


def get_product_name(params):
    # no params for check
    beging_t = datetime.now()
    try:
        response_list = []
        for idx, product in enumerate(SPU_LIST):
            response_list.append({'id': idx + 1, 'name': product})
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)
    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}\ttime_cost:{}'.format(response_list, cost))
    rlt_json = {
        'code': 200,
        'data': response_list,
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def get_sku_list(params):

    try:
        assert 'spu' in params
    except:
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)
    spu = params['spu'][0]
    beging_t = datetime.now()
    try:
        response_list = []
        for idx, sku in enumerate(SKU_DICT[spu]):
            response_list.append({'id': idx+1, 'name': sku})
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)
    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}\ttime_cost:{}'.format(response_list, cost))
    rlt_json = {
        'code': 200,
        'data': response_list,
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def get_tag_list(params):

    try:
        assert 'domain' in params
    except:
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)
    domain = params['domain'][0]
    beging_t = datetime.now()
    try:
        response_list = []
        for idx, tag in enumerate(TAG_LIST):
            response_list.append({'id': idx + 1, 'name': tag})
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)
    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}\ttime_cost:{}'.format(response_list, cost))
    rlt_json = {
        'code': 200,
        'data': response_list,
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def get_picture(params):
    try:
        assert 'spu' in params
        assert 'sku' in params
        assert 'startTime' in params
        assert 'endTime' in params
        assert 'domain' in params
    except:
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    spu = params['spu']
    sku = params['sku'] if params['sku'] != '全部' else ""
    domain = params['domain']

    s_y, s_m, s_d = timestamp2date(params['startTime'])
    e_y, e_m, e_d = timestamp2date(params['endTime'])

    query = {
        'spu': spu,
        'sku': {'$regex': sku},
        "comment_time": {"$gte": datetime(int(s_y), int(s_m), int(s_d)), "$lte": datetime(int(e_y), int(e_m), int(e_d))}
    }

    beging_t = datetime.now()
    comments_docs = mongodb_client.db['comments'].find(query)
    summary_tag_dict = gensummary.Summary.get_statistical_analysis(list(comments_docs))

    response_dict = {}
    for tag in TAG_LIST:
        response_dict[tag] = {}
        response_dict[tag]['pos'] = ''
        response_dict[tag]['neg'] = ''

    try:
        for tag in response_dict:
            if tag in summary_tag_dict:
                response_dict[tag]['pos'] = summary_tag_dict[tag]['pos'] if summary_tag_dict[tag]['pos'] != 0 else ''
                response_dict[tag]['neg'] = summary_tag_dict[tag]['neg'] if summary_tag_dict[tag]['neg'] != 0 else ''
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)
    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}\ttime_cost:{}'.format(response_dict, cost))

    rlt_json = {
        'code': 200,
        'data': response_dict,
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def get_picture_trends(params):
    try:
        assert 'spu' in params
        assert 'sku' in params
        assert 'startTime' in params
        assert 'endTime' in params
        assert 'domain' in params
    except:
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    beging_t = datetime.now()

    spu = params['spu']
    sku = params['sku'] if params['sku'] != '全部' else ""
    domain = params['domain']

    start_time = int(params['startTime'])
    end_time = int(params['endTime'])
    s_y, s_m, s_d = timestamp2date(params['startTime'])
    e_y, e_m, e_d = timestamp2date(params['endTime'])

    difference = end_time - start_time + ONE_DAY_LENGTH
    week_shift = difference / ONE_WEEK_LENGTH

    query_list = []
    if week_shift == 6:
        for i in range(6):
            cur_start_time = start_time + i * ONE_WEEK_LENGTH
            cur_s_y, cur_s_m, cur_s_d = timestamp2date(cur_start_time)

            cur_end_time = cur_start_time + ONE_DAY_LENGTH * 6
            cur_e_y, cur_e_m, cur_e_d = timestamp2date(cur_end_time)

            query_list.append({
                'spu': spu,
                'sku': {'$regex': sku},
                "comment_time": {"$gte": datetime(int(cur_s_y), int(cur_s_m), int(cur_s_d)),
                                 "$lte": datetime(int(cur_e_y), int(cur_e_m), int(cur_e_d))}
            })
    else:
        cur_s_y = int(s_y)

        for i in range(6):
            cur_s_m = int(s_m) + i

            if cur_s_m < 12:
                cur_e_y = cur_s_y = int(s_y)  # <- 年
                cur_e_m = cur_s_m + 1

            elif cur_s_m == 12:
                cur_e_y = int(s_y) + 1
                cur_e_m = 1
            else:
                cur_s_y = int(s_y) + 1
                cur_s_m = cur_s_m % 12
                cur_e_m = cur_s_m + 1
                cur_e_y = cur_s_y

            query_list.append({
                'spu': spu,
                'sku': {'$regex': sku},
                "comment_time": {"$gte": datetime(int(cur_s_y), int(cur_s_m), 1),  # <- 每个月1号
                                 "$lt": datetime(int(cur_e_y), int(cur_e_m), 1)}
            })

    response_dict = {}
    for tag in TAG_LIST:
        response_dict[tag] = {}
        response_dict[tag]['pos'] = []
        response_dict[tag]['neg'] = []
    time_list = []
    for idx, query in enumerate(query_list):
        comments_docs = mongodb_client.db['comments'].find(query)
        summary_tag_dict = gensummary.Summary.get_statistical_analysis(list(comments_docs))
        time_for_response = query['comment_time']['$gte'].strftime("%Y-%m-%d")
        time_list.append(time_for_response)
        try:
            for tag in response_dict:
                if tag in summary_tag_dict:
                    response_dict[tag]['pos'].append(summary_tag_dict[tag]['pos'] if summary_tag_dict[tag]['pos'] != 0 else '')
                    response_dict[tag]['neg'].append(summary_tag_dict[tag]['neg'] if summary_tag_dict[tag]['neg'] != 0 else '')
                else:
                    response_dict[tag]['pos'].append('')
                    response_dict[tag]['neg'].append('')
        except Exception as e:
            rlt_json = {
                'code': 500,
                'error': e
            }
            return json.dumps(rlt_json, ensure_ascii=True)

    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}\ttime_cost:{}'.format(response_dict, cost))
    rlt_json = {
        'code': 200,
        'data': response_dict,
        'time_cost': cost,
        'time_list': time_list
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def get_comments(params):
    try:
        assert 'spu' in params
        assert 'sku' in params
        assert 'polarity' in params
        assert 'tag' in params
        assert 'startTime' in params
        assert 'endTime' in params
        assert 'pageNo' in params
        assert 'pageSize' in params
    except Exception as e:
        logger.error(e)
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    spu = params['spu']
    sku = params['sku']
    sku = sku if sku != '全部' else ""
    polarity = params['polarity']
    tag = params['tag']
    s_y, s_m, s_d = timestamp2date(params['startTime'])
    e_y, e_m, e_d = timestamp2date(params['endTime'])

    page_no = int(params['pageNo'][0])
    page_no = page_no if page_no != 0 else 1

    page_size = int(params['pageSize'][0])

    query = {
        'spu': spu,
        'sku': {'$regex': sku},
        'pairs': {'$elemMatch': {'tag': tag, 'polarity': int(polarity)}},
        "comment_time": {"$gte": datetime(int(s_y), int(s_m), int(s_d)), "$lte": datetime(int(e_y), int(e_m), int(e_d))}
    }

    beging_t = datetime.now()
    total_count = mongodb_client.db['comments'].find(query).sort('comment_time', pymongo.DESCENDING).count()
    total_pagecount = int((total_count + page_size - 1) / page_size)  # 向上取整
    page_no = page_no if page_no <= total_pagecount else total_pagecount
    skip_page = (page_no - 1) * page_size if page_no > 0 else 0
    query_res = mongodb_client.db['comments'].find(query).sort('comment_time', pymongo.DESCENDING)\
        .skip(skip_page)\
        .limit(page_size)

    logger.info("全部记录数:{}\n全部页数:{}\n当前页:{}\n每页记录数:{}\n".format(total_count, total_pagecount, page_no, page_size))

    try:
        response_list = []
        for doc in query_res:
            cur_dict = {}
            cur_dict['spu'] = doc['spu']
            cur_dict['sku'] = doc['sku']
            cur_dict['domain'] = doc['domain']
            cur_dict['clean_comment'] = doc['clean_comment']
            cur_dict['comment_time'] = doc['comment_time']
            cur_dict['pairs'] = []
            cur_dict['comment_time'] = doc['comment_time'].strftime("%Y-%m-%d")
            if 'pairs' in doc:
                for pair in doc['pairs']:
                    if pair['tag'] == tag:
                        cur_dict['pairs'].append(pair)
                response_list.append(cur_dict)
    except Exception as e:
        logger.error(e)
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    cost = (datetime.now() - beging_t).total_seconds() * 1000

    logger.info('{}\ttime_cost:{}'.format(response_list, cost))

    rlt_json = {}
    rlt_json['code'] = 200
    rlt_json['data'] = {}
    page_dict = {}
    page_dict['pageNo'] = page_no
    page_dict['pageSize'] = page_size
    page_dict['totalCount'] = total_count
    page_dict['totalPageCount'] = total_pagecount
    rlt_json['data']['page'] = page_dict
    rlt_json['data']['items'] = response_list
    rlt_json['time_cost'] = cost

    return json.dumps(rlt_json, ensure_ascii=True)


def do_preprocess(params):
    try:
        assert 'domain' in params
        assert 'spu' in params
        assert 'startTime' in params
    except:
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    domain = params['domain'][0]
    spu = params['spu'][0]
    s_y, s_m, s_d = timestamp2date(params['startTime'][0])

    query = {
        'domain': domain,
        "comment_time": {"$gt": datetime(int(s_y), int(s_m), int(s_d))}
    }
    if spu != 'none':
        query['spu'] = spu

    comments_collection = mongodb_client.db['comments']
    beging_t = datetime.now()
    try:
        """数据预处理，并获取分词，词性，依存结果,结果存回数据库"""
        res_str = processor.process_raw_data_and_save_in_db(comments_collection, query)
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}\ttime_cost:{}'.format(res_str, cost))
    rlt_json = {
        'code': 200,
        'status': 'success',
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def build(params):
    try:
        assert 'domain' in params
    except:
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    domain = params['domain'][0]

    beging_t = datetime.now()

    resources = utils.load_resources(domain)
    try:
        """抽取aspect并过滤，phrase的发现，最终aspect表"""
        aspectext = genaspect.AspectExtractor(resources)
        aspect_for_filter = aspectext.extract_aspect_by_opinion_window()
        """构建带有极性的pair，并将结果存回db，这里需要连上线上数据中心"""
        res_str, different_dict = build_pair_flow.build_pair(domain, mongodb_client, resources, aspect_for_filter)
    except Exception as e:
        logger.error(e)
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}\ttime_cost:{}'.format(res_str, cost))

    rlt_json = {
        'code': 200,
        'data': res_str,
        'different_dict': different_dict,
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def batch_infer(params):
    try:
        assert 'domain' in params
        assert 'spu' in params
    except Exception as e:
        logger.error(e)
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    domain = params['domain'][0]
    spu = params['spu'][0]
    beging_t = datetime.now()

    try:
        build_pairs_query_res = mongodb_client.db['opinion_build_pairs'].find_one({'domain': domain})
        pair_polarity = build_pairs_query_res['pair_polarity']
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        logger.error('mongodb io error:{}'.format(e))
        return json.dumps(rlt_json, ensure_ascii=True)

    comment_query = {
        'domain': domain
    }
    if spu != 'none':
        comment_query['spu'] = spu

    try:
        comment_query_res = mongodb_client.db['comments'].find(comment_query)
        comment_docs = list(comment_query_res)
        comment_collection = mongodb_client.db['comments']
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        logger.error('mongodb io error:{}'.format(e))
        return json.dumps(rlt_json, ensure_ascii=True)

    resources = utils.load_resources(domain)
    resources.pair_polarity = pair_polarity
    try:
        """抽取（aspect，opinion，polarity，clue）元组"""
        aopair = genaopair.AOPair(
            comment_docs,
            pair_polarity,
            resources,
            )
        res_str = aopair.inference_res_save_in_db(comment_collection)
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}\ttime_cost:{}'.format(res_str, cost))
    rlt_json = {
        'code': 200,
        'status': 'success',
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def infer_one(params):
    try:
        assert 'text' in params
    except:
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    text = params['text'][0]

    beging_t = datetime.now()
    clean_text = processor.clean(text)

    word_segments, pos_tag = processor.get_pos_seg_praser(clean_text)
    logger.info("word_segments:{}, pos_tag:{}".format(word_segments, pos_tag))

    pair_polarity = res_for_one.pair_polarity

    try:
        """抽取（aspect，opinion，polarity，clue）元组"""
        aopair = genaopair.AOPair(
            word_segments,
            pair_polarity,
            res_for_one,
        )
        res_str = aopair.get_infer_one_res()
    except Exception as e:
        logger.error(e)
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}\t{}\ttime_cost:{}'.format(text, res_str, cost))
    rlt_json = {
        'code': 200,
        'raw_comment': text,
        'data': res_str,
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def save_raw_data(params):
    try:
        assert 'filePath' in params
        assert 'domain' in params
    except:
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    beging_t = datetime.now()
    file_path = params['filePath'][0]
    domain = params['domain'][0]

    try:
        latest_comment = mongodb_client.db['comments'].find().sort('comment_time', pymongo.DESCENDING).limit(1)[0]
        latest_datetime = latest_comment['comment_time']
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        logger.error('mongodb io error:{}'.format(e))
        return json.dumps(rlt_json, ensure_ascii=True)

    try:
        r = requests.get(file_path)
        raw_data = r.content.decode('utf-8').split('\n')[:-1]
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        logger.error('get crawler data error:{}'.format(e))
        return json.dumps(rlt_json, ensure_ascii=True)

    new_spu_sku = {}
    insert_comments_query = []
    def get_formatted_time(comment_time):
        timearray = time.localtime(int(comment_time) / 1000)
        formatted_time = datetime.strptime(time.strftime("%Y-%m-%d", timearray), "%Y-%m-%d")
        return formatted_time

    for line in raw_data:
        if len(line) == 0:
            continue
        doc = json.loads(line)
        spu = doc['productName']
        sku = doc['skuInfo']
        if spu not in new_spu_sku:
            new_spu_sku[spu] = set()
        new_spu_sku[spu].add(sku)
        comment_list = doc['evaluationList']
        product_id = doc['productId']
        buyer_nick = doc['buyerNick']
        order_id = doc['buyerOrder']
        source = 'tmall'
        if '手表' in spu:
            for comment in comment_list:
                comment_time = comment['evaluationTime']
                comment_time = get_formatted_time(comment_time)
                comment_content = comment['evaluationContent']
                #按照时间过滤
                if comment_time <= latest_datetime:
                    continue
                year, month, day = comment_time.year, comment_time.month, comment_time.day
                insert_comments_query.append({
                    'domain': domain,
                    'spu': spu,
                    'sku': sku,
                    'order_id': order_id,
                    'raw_comment': comment_content,
                    'source': source,
                    'comment_time': datetime(int(year), int(month), int(day))
                })
    try:
        """插入新评论"""
        if insert_comments_query:
            r = mongodb_client.db['comments'].insert_many(insert_comments_query)
            logger.info('插入新评论{}条'.format(len(r.inserted_ids)))
        else:
            logger.info('没有新评论')
        # """更新spu"""
        # spu_list = mongodb_client.db['opinion_resources'].find_one({'doc_type': 'spu_list'})['lexicon']
        # for spu in new_spu_sku:
        #     spu_list.append(spu)
        # spu_list = list(set(spu_list))
        # mongodb_client.db['opinion_resources'].update({'doc_type': 'spu_list'}, {'$set': {'lexicon': spu_list}}, upsert=True)
        # logger.info('更新spu成功')
        # """更新sku"""
        # sku_dict = mongodb_client.db['opinion_resources'].find_one({'doc_type': 'sku_list'})['lexicon']
        # for spu in new_spu_sku:
        #     sku_dict[spu] = list(new_spu_sku[spu])
        # mongodb_client.db['opinion_resources'].update({'doc_type': 'sku_list'}, {'$set': {'lexicon': sku_dict}}, upsert=True)
        # logger.info('更新sku成功')
    except Exception as e:
        logger.error(e)
        rlt_json = {
            'code': 500,
            'error': e
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('新数据插入数据库成功,time_cost:{}'.format(cost))
    rlt_json = {
        'code': 200,
        'status': 'success',
        'latestTime': int(latest_datetime.timestamp()) * 1000,
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


def get_csv(params):
    try:
        assert 'domain' in params
        assert 'startTime' in params
        assert 'endTime' in params
    except:
        rlt_json = {
            'code': 500,
            'error': 'error parameters:{}'.format(params)
        }
        return json.dumps(rlt_json, ensure_ascii=True)

    domain = params['domain'][0]
    startTime = params['startTime'][0]
    endTime = params['endTime'][0]

    beging_t = datetime.now()
    s_y, s_m, s_d = timestamp2date(startTime)
    e_y, e_m, e_d = timestamp2date(endTime)

    query = {
        "comment_time": {"$gte": datetime(int(s_y), int(s_m), int(s_d)), "$lte": datetime(int(e_y), int(e_m), int(e_d))}
    }
    try:
        docs = mongodb_client.db['comments'].find(query).sort('comment_time', pymongo.DESCENDING)
    except Exception as e:
        rlt_json = {
            'code': 500,
            'error': e
        }
        logger.error('mongodb io error:{}'.format(e))
        return json.dumps(rlt_json, ensure_ascii=True)

    csv_content = []
    csv_content.append(['spu', 'sku', 'tag', 'aspect', 'polarity', 'express', 'raw_comment', 'orderID', 'orderDate', 'source'])

    map_polarity = {1: '正向', -1: '负向', 2: '不确定', -2: '不确定'}

    for doc in docs:
        if 'pairs' not in doc:
            continue
        spu = doc['spu']
        sku = doc['sku']
        raw_comment = doc['raw_comment']
        order_id = '' if 'order_id' not in doc else doc['order_id']
        source = '' if 'source' not in doc else doc['source']
        pairs = doc['pairs']
        order_date = doc['comment_time'].strftime("%Y-%m-%d")
        for pair in pairs:
            tag = pair['tag']
            aspect = ','.join(pair['aspect'])
            polarity = map_polarity[pair['polarity']]
            express = pair['raw_express']
            csv_content.append([spu, sku, tag, aspect, polarity, express, raw_comment, order_id, order_date, source])

    cost = (datetime.now() - beging_t).total_seconds() * 1000
    logger.info('{}-{}数据共{}条,time_cost:{}'.format(timestamp2str(startTime), timestamp2str(endTime), len(csv_content), cost))

    rlt_json = {
        'code': 200,
        'data': csv_content,
        'time_cost': cost
    }
    return json.dumps(rlt_json, ensure_ascii=True)


if __name__ == "__main__":
    logger.info('starting front server...')
    load_spu_list()
    load_sku_list()
    load_tag_list('手表')
    load_resources_for_infer_one('手表')
    host = ''
    port = ''
    try:
        port = int(sys.argv[1])
    except:
        print('One parameter required: PORT.')
        exit()

    httpd = OMHTTPServer(host, port)
    #
    httpd.Register('/productName', get_product_name)
    httpd.Register('/tagList', get_tag_list)
    httpd.Register('/comments', get_comments)
    httpd.Register('/productSku', get_sku_list)
    httpd.Register('/picture', get_picture)
    httpd.Register('/pictureTrends', get_picture_trends)
    #
    httpd.Register('/preprocess', do_preprocess)
    httpd.Register('/build', build)
    httpd.Register('/batchInfer', batch_infer)
    httpd.Register('/inferOne', infer_one)
    httpd.Register('/saveRawData', save_raw_data)
    httpd.Register('/getCSV', get_csv)
    logger.info('running server...')
    httpd.serve_forever()
