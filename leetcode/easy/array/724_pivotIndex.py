# coding=utf-8

# https://leetcode-cn.com/problems/find-pivot-index/

class Solution(object):
  def pivotIndex(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    total = sum(nums)
    left_sum = 0
    for idx, n in enumerate(nums):
      if total - left_sum - n == left_sum:
        return idx
      left_sum += n
    return -1
