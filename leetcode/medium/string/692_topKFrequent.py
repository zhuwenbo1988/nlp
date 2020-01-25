# coding=utf-8

# https://leetcode-cn.com/problems/top-k-frequent-words/

class Solution(object):
  def topKFrequent(self, words, k):
    """
    :type words: List[str]
    :type k: int
    :rtype: List[str]
    """
    # 统计频次
    d = {}
    for w in words:
      if w in d:
        d[w] += 1
      else:
        d[w] = 1
    # 装桶
    b = [[] for _ in range(len(words)+1)]
    for w in d:
      n = d[w]
      b[n].append(w)
    # 取出来结果
    result = []
    for n in range(len(words), 0, -1):
      if not b[n]:
        continue
      if len(result) > k:
        break
      words = b[n]
      words.sort()
      result.extend(words)
    return result[:k]
