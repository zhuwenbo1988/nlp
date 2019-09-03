# -*- coding:utf-8 -*-

import os
import sys
import codecs

reload(sys)
sys.setdefaultencoding('utf-8')

def load_train_data(data_path, save_path, query_len_list,
                     response_len_list, pairs_count, turn="multi"):
    with codecs.open(save_path, "a", encoding="utf-8") as writer:
        with codecs.open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line:
                    continue
                lines = line.strip().split("\t")
                if len(lines) < 3:
                    continue
                pairs_count += 1
                label = str(lines[0])
                if turn == "multi":
                    query = " ".join(lines[1:-1])
                else:
                    query = lines[-2]
                response = lines[-1]
                length_query = len(query.strip().split())
                length_response = len(response.strip().split())
                query_len_list.append(length_query)
                response_len_list.append(length_response)
                result = (label + "\t" + query.strip() + "\t"
                          + response.strip() + "\n")
                writer.write(result)
    return query_len_list, response_len_list, pairs_count


def init_train_dataset(train_path, dev_path, test_path, save_path, turn):
    # 统计所有query的长度
    query_len_list = []
    # response长度
    response_len_list = []
    # 数据总数
    pairs_count = 0
    if os.path.exists(save_path):
        os.remove(save_path)
    # 加载豆瓣训练数据
    query_len_list, response_len_list, pairs_count = load_train_data(
        train_path, save_path, query_len_list, response_len_list, pairs_count, turn)
    # 加载验证数据
    query_len_list, response_len_list, pairs_count = load_train_data(
        dev_path, save_path, query_len_list, response_len_list, pairs_count, turn)
    # 加载测试数据
    query_len_list, response_len_list, pairs_count = load_train_data(
        test_path, save_path, query_len_list, response_len_list, pairs_count, turn)
    query_len_list = sorted(query_len_list)
    response_len_list = sorted(response_len_list)
    print("count: ", pairs_count)
    print("query_len", query_len_list[int(pairs_count * 0.95)])
    print("response_len", response_len_list[int(pairs_count * 0.95)])

    
if __name__ == "__main__":
    basedir = sys.argv[1]

    train_path = os.path.join(basedir, "douban_jieba/train.txt")
    dev_path = os.path.join(basedir, "douban_jieba/dev.txt")
    test_path = os.path.join(basedir, "douban_jieba/test.txt")
    # 保存数据集
    save_path = os.path.join(basedir, "matchpyramid/dataset.txt")
    matchpyramid_dir, _ = os.path.split(save_path)
    if not os.path.exists(matchpyramid_dir):
        os.mkdir(matchpyramid_dir)
    # 生成数据集
    # 多轮,否则只取上下文的最后一句
    turn = "multi"
    init_train_dataset(train_path, dev_path, test_path, save_path, turn)
    print("done!")