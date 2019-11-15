# coding=utf-8

'''

三指针

'''

# https://leetcode-cn.com/problems/sort-colors

class Solution(object):
  def sortColors(self, nums):
    """
    :type nums: List[int]
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    s = 0
    m = 0
    e = len(nums) - 1
    while m <= e:
      if nums[m] == 2:
        tmp = nums[e]
        nums[e] = nums[m]
        nums[m] = tmp
        e += -1
      elif nums[m] == 0:
        nums[s] = 0
        if m > s:
          nums[m] = 1
        s += 1
        m += 1
      else:
        m += 1
