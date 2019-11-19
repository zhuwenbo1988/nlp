# coding=utf-8

'''

先统计频次

将数组中的元素按照出现频次进行分组，即出现频次为 iii 的元素存放在第 iii 个桶。最后，从桶中逆序取出前 kkk 个元素。

'''

# https://leetcode-cn.com/problems/top-k-frequent-elements/

class Solution(object):
  def topKFrequent(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[int]
    """
    d = {}
    for n in nums:
      d[n] = d.get(n, 0) + 1

    b = [[] for i in range(len(nums)+1)]
    for key, v in d.iteritems():
      b[v].append(key)

    result = []
    for i in range(len(nums), -1, -1):
      if b[i]:
        result.extend(b[i])
      if len(result) >= k:
        break
    return result[:k]
