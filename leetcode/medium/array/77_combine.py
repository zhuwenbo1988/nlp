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
    def find(k, other, tmp):
      if k == 0:
        result.append(tmp)
        return
      for i in range(len(other)):
        find(k-1, other[i+1:], tmp + [other[i]])
    find(k, nums, [])
    return result
