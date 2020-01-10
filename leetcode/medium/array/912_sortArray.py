# coding=utf-8

'''

快排

'''

# https://leetcode-cn.com/problems/sort-an-array/

class Solution(object):
  def sortArray(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    l = []
    l.insert(0, len(nums)-1)
    l.insert(0, 0)
    while l:
      left = l[0]
      del l[0]
      right = l[0]
      del l[0]
      cut = self.sort(nums, left, right)
      if left < cut-1:
        l.insert(cut-1)
        l.insert(left)
      if right > cut+1:
        l.insert(right)
        l.insert(cut+1)
    return nums

  def sort(self, nums, left, right):
    val = nums[left]
    while left < right:
      while left < right and nums[right] > val:
        right += -1
      nums[left] = nums[right]
      while left < right and nums[left] <= val:
        left += 1
      nums[right] = nums[left]
    # left=right
    nums[left] = val
    return left
