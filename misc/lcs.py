# coding=utf-8

def get_lcsubstr(s1, s2):
    """longest common substring"""
    score = 0.0
    len1 = len(s1)
    len2 = len(s2)
    if not len1 or not len2:
        return score
    m = [[0 for i in range(len2 + 1)] for j in range(len1 + 1)]
    mmax = 0
    for i in range(len1):
        for j in range(len2):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
    min_len = len1 if len1 < len2 else len2
    score = 1.0 * mmax / min_len
    return score
