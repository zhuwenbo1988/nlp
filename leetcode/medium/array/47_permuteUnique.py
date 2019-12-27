# coding=utf-8

'''

与46的唯一区别就是增加了去重

'''

# https://leetcode-cn.com/problems/permutations-ii/

class Solution(object):
  def permuteUnique(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    d = {}
    result = []
    def p(other, curr):
      if not other:
        key = ''.join([str(i) for i in curr])
        if key not in d:
          result.append(curr)
          d[key] = 1
        return
      for i in range(len(other)):
        p(other[:i] + other[i+1:], curr + [other[i]])
    p(nums, [])
    return result
