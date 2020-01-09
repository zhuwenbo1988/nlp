# coding=utf-8

'''

设置一个index，表示非0数的个数，循环遍历数组

'''

# https://leetcode-cn.com/problems/move-zeroes/

class Solution(object):
  def moveZeroes(self, nums):
    """
    :type nums: List[int]
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    if not nums:
      return
    if len(nums) == 1:
      return
    curr = 0
    for i in range(len(nums)):
      if nums[i] != 0:
        nums[curr] = nums[i]
        if curr != i:
          nums[i] = 0
        curr += 1
