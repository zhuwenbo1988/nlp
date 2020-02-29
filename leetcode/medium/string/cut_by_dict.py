# coding=utf-8

'''

https://www.cnblogs.com/hiyoung/p/9687423.html

基于字典的分词方法

'''

d = {}
d[u'研究'] = 1
d[u'生命'] = 1
d[u'起源'] = 1
d[u'研究生'] = 1

s = u'研究生命的起源'

window_size = max([len(w) for w in d])

def forward_cut(raw_s, word_dict):
    result = []
    head = 0
    size = len(raw_s)
    while size > head:
        for i in range(head+window_size, head, -1):
            token = raw_s[head:i]
            if token in word_dict:
                head = i-1
                break
        head += 1
        result.append(token)
    return result

def backward_cut(raw_s, word_dict):
    result = []
    tail = len(raw_s)
    global window_size
    window_size = min(window_size, tail)
    while tail > 0:
        for i in range(tail-window_size, tail):
            token = raw_s[i:tail]
            if token in word_dict:
                tail = i+1
                break
        tail += -1
        result.append(token)
    result = result[::-1]
    return result


def bi_cut(raw_s, word_dict):
    f_tokens = forward_cut(raw_s, word_dict)
    b_tokens = backward_cut(raw_s, word_dict)
    if len(f_tokens) < len(b_tokens):
        return f_tokens
    if len(f_tokens) > len(b_tokens):
        return b_tokens
    # 词数量相等
    # 前向结果与后向结果相等
    if f_tokens == b_tokens:
        return f_tokens
    # 分词结果不同，返回其中单字数量较少的那个
    f_n = len([w for w in f_tokens if len(w)==1])
    b_n = len([w for w in b_tokens if len(w)==1])
    if f_n > b_n:
        return b_tokens
    elif f_n < b_n:
        return f_tokens
    else:
        return f_tokens

print ' '.join(forward_cut(s, d))
print ' '.join(backward_cut(s, d))
print ' '.join(bi_cut(s, d))