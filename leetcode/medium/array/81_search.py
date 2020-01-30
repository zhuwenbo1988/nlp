# coding=utf-8

'''

数组的元素可以重复
中点不能于nums[0]进行比较，因为全局已经不具有严格的递增特性

'''

# https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/

class Solution(object):
  def search(self, nums, target):
    if not nums:
      return False
    left, right = 0, len(nums) - 1
    while left <= right:
      mid = (left + right) // 2
      v = nums[mid]
      if target == v:
        return True
      # [left, mid]升序
      if v > nums[left]:
        # target位于[left, mid]
        if target < v and target >= nums[left]:
          right = mid - 1
        else: # ççç
          left = mid + 1
      elif v < nums[left]: # [mid, right]升序
        # target位于[mid, right]
        if target > v and target <= nums[right]:
          left = mid + 1
        else: # target位于[left, mid]
          right = mid - 1
      else:
        left += 1
    return False
