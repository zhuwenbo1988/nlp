# coding=utf-8

# https://leetcode-cn.com/problems/find-peak-element/

class Solution(object):
  def findPeakElement(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    l = 0
    r = len(nums)-1
    while l < r:
      mid = (l+r) // 2
      if nums[mid] > nums[mid-1] and nums[mid] > nums[mid+1]:
        return mid
      if nums[mid] < nums[mid+1]:
        l = mid + 1
      else:
        r = mid - 1
    return l
