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
    def find(other, k, tmp):
      if k == 0:
        result.append(tmp)
        return
      for i in range(len(other)-k+1):
        find(other[i+1:], k-1, tmp + [other[i]])
    find(nums, k, [])
    return result
