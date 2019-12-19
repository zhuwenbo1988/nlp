# coding=utf-8

'''

回溯算法

'''

# https://leetcode-cn.com/problems/combination-sum

class Solution(object):
  def combinationSum(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    result = []
    tmp = []
    def find(i, curr_sum, tmp):
      if curr_sum == target:
        result.append(tmp)
        return
      if curr_sum > target or i == len(candidates):
        return
      find(i, curr_sum + candidates[i], tmp + [candidates[i]])
      find(i+1, curr_sum, tmp)
    find(0, 0, tmp)
    return result
