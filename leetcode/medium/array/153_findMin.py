# coding=utf-8

'''

前面一堆０，后面一堆１，然后寻找第一个１的二分问题

也可借鉴34题，逐渐逼近目标点

'''

# https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/

class Solution(object):
  def findMin(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    l = 0
    r = len(nums)-1
    # 关键点 - 小于
    while l < r:
      mid = (r + l) // 2
      # 关键点 - nums[-1]
      if nums[mid] < nums[-1]:
        # 关键点 - r不是mid-1
        r = mid
      else:
        l = mid+1
    return nums[l]
