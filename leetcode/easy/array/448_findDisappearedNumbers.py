# coding=utf-8

# https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/

class Solution(object):
  def findDisappearedNumbers(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    for i in nums:
      if nums[abs(i)-1] > 0:
        nums[abs(i)-1] *= -1
    result = []
    for i in range(len(nums)):
      if nums[i] < 0:
        continue
      result.append(i+1)
    return result
