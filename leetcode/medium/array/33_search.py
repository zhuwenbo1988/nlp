# coding=utf-8

'''

方法就是二分查找，但是很烧脑

'''

# https://leetcode-cn.com/problems/search-in-rotated-sorted-array/

class Solution(object):
  def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    left = 0
    right = len(nums) - 1
    # 小于等于
    while left <= right:
      mid = (left + right) // 2
      if nums[mid] == target:
        return mid
      v = nums[mid]
      '''
      中点位于左边升序数组
        目标值小于中点,目标值大于等于左端点 - 目标值在左边升序数组的中点的左边
        否则 - 目标值位于左边升序数组的中点的右边或右边升序数组
      中点位于右边升序数组
        目标值大于中点，目标值小于左端点 - 目标值在右边升序数组的中点的右边
        否则 - 目标值位于左边升序数组或右边升序数组的中点的左边
      '''
      if v >= nums[0]:
        if target < v and target >= nums[0]:
          right = mid - 1
        else:
          left = mid + 1
      else:
        if target > v and target < nums[0]:
          left = mid + 1
        else:
          right = mid - 1
    return -1