# coding=utf-8

# https://leetcode-cn.com/problems/permutations-ii/

class Solution(object):
  def permuteUnique(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    d = {}
    result = []
    def p(curr, other):
      if not other:
        key = ''.join([str(i) for i in curr])
        if key not in d:
          result.append(curr)
          d[key] = 1
        return
      for i in range(len(other)):
        p(curr + [other[i]], other[:i] + other[i+1:])
    p([], nums)
    return result
