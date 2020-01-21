# coding=utf-8

# https://leetcode-cn.com/problems/increasing-subsequences/

class Solution(object):
  def findSubsequences(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    result = {}
    def find(other, tmp):
      key = '-'.join([str(i) for i in tmp])
      if len(tmp) > 1 and key not in result:
        result[key] = tmp
      for i in range(len(other)):
        if not tmp:
          find(other[i+1:], [other[i]])
          continue
        if other[i] >= tmp[-1]:
          find(other[i+1:], tmp + [other[i]])
    find(nums, [])
    return result.values()
