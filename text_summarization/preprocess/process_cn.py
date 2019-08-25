# -*- coding:utf-8 -*-
# @Copyright 2019 mobvoi Inc.
# @Author   :wbzhu


from opencc import OpenCC
import re


def convert_tcn2scn(text):
    """繁体中文转简体中文
    使用opencc实现
    https://github.com/yichen0831/opencc-python

    Keyword arguments:
    text  --  需要处理的文本,type需要是unicode
    """
    cc = OpenCC('t2s')
    converted = cc.convert(text)
    return converted


def convert_strQ2B(text):
    """全角转半角

    Keyword arguments:
    text  --  需要处理的文本,type需要是unicode
    """
    rstring = ""
    for uchar in text:
        inside_code = ord(uchar)
        # 全角空格直接转换
        if inside_code == 12288:
            inside_code = 32
        # 全角字符（除空格）根据关系转化
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring


def detect_nonstop_punctuation(text, punctuation):
    """把不是断句意思的标点符号替换成特殊的标志

    Keyword arguments:
    text  --  需要处理的文本,type需要是unicode
    punctuation  --  指定的标点符号列表,每个符号需要是unicode
    """
    for p in punctuation:
        text = text.replace(p, ' NP ')
    return text


def detect_long_stop_punctuation(text, punctuation):
    """把是断句意思的标点符号替换成特殊的标志

    Keyword arguments:
    text  --  需要处理的文本,type需要是unicode
    punctuation  --  指定的标点符号,type需要是unicode
    """
    for p in punctuation:
        text = text.replace(p, ' LSP ')
    return text


def detect_short_stop_punctuation(text, punctuation):
    """把是断句意思的标点符号替换成特殊的标志

    Keyword arguments:
    text  --  需要处理的文本,type需要是unicode
    punctuation  --  指定的标点符号,type需要是unicode
    """
    for p in punctuation:
        text = text.replace(p, ' SSP ')
    return text


def detect_datestr(text):
    text = re.sub(u"\d+年", ' D ', text)
    text = re.sub(u"\d+月", ' D ', text)
    text = re.sub(u"\d+日", ' D ', text)
    return text


def detect_number(text):
    """把阿拉伯数字替换成特殊的标志

    Keyword arguments:
    text  --  需要处理的文本,type需要是unicode
    """
    text = re.sub('[\d]+', ' N ', text)
    return text


def detect_enword(text):
    """把英文单词替换成特殊的标志

    Keyword arguments:
    text  --  需要处理的文本,type需要是unicode
    """
    text = re.sub('[a-zA-Z]+', ' E ', text)
    return text


def detect_bracket_content(text):
    """监测文本中的括号

    Keyword arguments:
    text  --  需要处理的文本,type需要是unicode
    """
    text = re.sub("\(.*?\)", ' B ', text)
    text = re.sub("\[.*?\]", ' B ', text)
    text = re.sub("【.*?】", ' B ', text)
    return text