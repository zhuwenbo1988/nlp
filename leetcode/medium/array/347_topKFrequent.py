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
    # 一定要先统计频次
    freq_dict = {}
    for n in nums:
      freq_dict[n] = freq_dict.get(n, 0) + 1
    # 桶排序
    b = [[] for i in range(len(nums)+1)]
    for number, freq in freq_dict.iteritems():
      b[freq].append(number)
    result = []
    for i in range(len(nums), -1, -1):
      if b[i]:
        result.extend(b[i])
      if len(result) >= k:
        break
    return result[:k]
