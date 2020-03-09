# coding=utf-8

'''

单调栈经典题目

'''

# https://leetcode-cn.com/problems/next-greater-element-ii/

class Solution(object):
  def nextGreaterElements(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    stack = []
    result = [-1]*len(nums)
    for i in range(len(nums)*2):
      j = i % len(nums)
      while stack and nums[stack[0]] < nums[j]:
        result[stack[0]] = nums[j]
        del stack[0]
      stack.insert(0, j)
    return result