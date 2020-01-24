# coding=utf-8

# https://leetcode-cn.com/problems/sort-characters-by-frequency/

class Solution(object):
  def frequencySort(self, s):
    """
    :type s: str
    :rtype: str
    """
    freq_dict = {}
    for c in s:
      if c in freq_dict:
        freq_dict[c] += 1
      else:
        freq_dict[c] = 1

    n = len(s)
    bucket = [[] for _ in range(n+1)]
    for c in freq_dict:
      freq = freq_dict[c]
      bucket[freq].append(c)

    result = []
    for i in range(n, 0, -1):
      if not bucket[i]:
        continue
      for c in bucket[i]:
        result.extend([c] * i)
    return ''.join(result)
