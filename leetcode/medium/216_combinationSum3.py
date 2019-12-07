# coding=utf-8

# https://leetcode-cn.com/problems/combination-sum-iii/

class Solution(object):
  def combinationSum3(self, k, n):
    """
    :type k: int
    :type n: int
    :rtype: List[List[int]]
    """
    nums = [i for i in range(1, 10)]
    result = []
    def find(tmp, other):
      if len(tmp) == k and sum(tmp) == n:
        result.append(tmp)
      if len(tmp) > k:
        return
      if sum(tmp) > n:
        return
      for i in range(len(other)):
        find(tmp + [other[i]], other[i+1:])
    find([], nums)
    return result
