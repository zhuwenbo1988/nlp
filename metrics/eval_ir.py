# coding=utf-8

# 计算map
def cla_MAP(predicts):
    average_precision = 0
    # 遍历标准结果
    for i, query in enumerate(predicts.keys()):
        precision = 0
        index = 0
        count = 0
        for j, pairs in enumerate(predicts[query]):
            index += 1
            # 检查是否匹配
            label = pairs[-1]
            if label == 1:
                count += 1
                precision += float(count) / index
        if len(predicts[query]) == 0 or count == 0:
            continue
        average_precision += float(precision)/count
    return float(average_precision)/len(predicts.keys())

# 计算P@1值
def cal_P1(predicts):
    count = 0
    # 遍历标准结果
    for query in predicts.keys():
        # 检查是否匹配
        label = predicts[query][0][-1]
        if label == 1:
                count += 1
    return float(float(count) / len(predicts.keys()))
