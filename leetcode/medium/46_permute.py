# coding=utf-8

# 回溯算法

# https://leetcode-cn.com/problems/permutations/

class Solution(object):
  def permute(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    result = []
    def back(other, ans):
      if not other:
        result.append(ans)
        return
      for i in range(len(other)):
        back(other[:i] + other[i+1:], ans + [other[i]])
    back(nums, [])
    return result
