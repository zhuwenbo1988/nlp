# coding=utf-8

'''

超时

'''

# https://leetcode-cn.com/problems/combination-sum-ii/

class Solution(object):
  def combinationSum2(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    result = []
    def add(curr_sum, other, tmp):
      if curr_sum == target and sorted(tmp) not in result:
        result.append(sorted(tmp))
        return
      if curr_sum > target or not other:
        return
      for i in range(len(other)):
        add(curr_sum + other[i], other[:i] + other[i+1:], tmp + [other[i]])
    add(0, candidates, [])
    return result
