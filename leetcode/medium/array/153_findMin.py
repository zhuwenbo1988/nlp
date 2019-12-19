# coding=utf-8

# https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/

class Solution(object):
  def findMin(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    l = 0
    r = len(nums)-1
    while l < r:
      mid = (r + l) // 2
      if nums[mid] < nums[-1]:
        r = mid
      else:
        l = mid+1
    return nums[l]
