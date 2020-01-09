# coding=utf-8

# https://leetcode-cn.com/problems/two-sum/

class Solution(object):
  def twoSum(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    m = {}
    for idx, n in enumerate(nums):
      if (target - n) in m:
        return (m[target-n], idx)
      m[n] = idx
    return
