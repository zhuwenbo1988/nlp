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
    def find(i, tmp):
      if sum(tmp) == target:
        result.append(tmp)
        return
      if sum(tmp) > target or i == len(candidates):
        return
      find(i, tmp + [candidates[i]])
      find(i+1, tmp)
    find(0, tmp)
    return result
