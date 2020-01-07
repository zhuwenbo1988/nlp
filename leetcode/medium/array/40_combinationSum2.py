# coding=utf-8

# https://leetcode-cn.com/problems/combination-sum-ii/

class Solution(object):
  def combinationSum2(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    candidates.sort()
    result = []
    def add(other, tmp):
      if sum(tmp) == target and sorted(tmp) not in result:
        result.append(sorted(tmp))
        return
      if sum(tmp) > target or not other:
        return
      for i in range(len(other)):
        if i > 0 and other[i] == other[i-1]:
            continue
        add(other[i+1:], tmp + [other[i]])
    add(candidates, [])
    return result