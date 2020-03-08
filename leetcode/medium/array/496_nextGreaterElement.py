# coding=utf-8

'''

单调栈经典题目

'''

# https://leetcode-cn.com/problems/next-greater-element-i/

class Solution(object):
  def nextGreaterElement(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    map = {}
    stack = []
    for i in range(len(nums2)):
      while stack and stack[0] < nums2[i]:
        map[stack[0]] = nums2[i]
        del stack[0]
      stack.insert(0, nums2[i])
    result = []
    for i in range(len(nums1)):
      if nums1[i] in map:
        result.append(map[nums1[i]])
      else:
        result.append(-1)
    return result
