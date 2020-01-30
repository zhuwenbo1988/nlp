# coding=utf-8

# https://leetcode-cn.com/problems/combinations/

class Solution(object):
  def combine(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: List[List[int]]
    """
    result = []
    nums = [i for i in range(1, n+1)]
    def find(tmp, other):
      if len(tmp) == k:
        result.append(tmp)
      for i in range(len(other)):
        find(tmp + [other[i]], other[i+1:])
    find([], nums)
    return result