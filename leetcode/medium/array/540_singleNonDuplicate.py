# coding=utf-8



# https://leetcode-cn.com/problems/single-element-in-a-sorted-array/

class Solution(object):
  def singleNonDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    s = 0
    e = len(nums)-1
    while s < e:
      mid = (s+e) // 2
      if mid % 2 == 1:
        mid += -1
      if nums[mid] == nums[mid+1]:
        s = mid + 2
      else:
        e = mid
    # s==e
    return nums[s]
