# coding=utf-8

'''

我们用三个指针来分别追踪0的最右边界，2的最左边界和当前考虑的元素

0，1，2 排序。一次遍历，如果是0，则移动到表头，如果是2，则移动到表尾，不用考虑1

'''

# https://leetcode-cn.com/problems/sort-colors

class Solution(object):
  def sortColors(self, nums):
    """
    :type nums: List[int]
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    p0 = 0
    curr = 0
    p2 = len(nums) - 1
    while curr <= p2:
      if nums[curr] == 2:
        tmp = nums[p2]
        nums[p2] = nums[curr]
        nums[curr] = tmp
        p2 += -1
      elif nums[curr] == 0:
        tmp = nums[p0]
        nums[p0] = nums[curr]
        nums[curr] = tmp
        p0 += 1
        curr += 1
      else:
        curr += 1
