# coding=utf-8

'''

利用二分思想先找其左边界，再找其右边界即可，注意找左边界的时候，由右侧逼近；找右边界的时候，由左侧逼近

'''

# https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/

class Solution(object):
  def searchRange(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    i = 0
    j = len(nums) - 1
    # 不是 <=
    while i < j:
      idx = (i + j) // 2
      v = nums[idx]
      if v >= target:
        j = idx
      else:
        i = idx + 1
    left = -1
    if nums:
      if nums[i] == target:
        left = i
    i = 0 
    j = len(nums) - 1
    # 不是 <=
    while i < j:
      # 重点 +1
      idx = (i + j) // 2 + 1
      v = nums[idx]
      if v <= target:
        i = idx
      else:
        j = idx - 1
    right = -1
    if nums:
      if nums[i] == target:
        right = i
    return [left, right]
