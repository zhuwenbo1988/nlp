# coding=utf-8

# https://leetcode-cn.com/problems/product-of-array-except-self/

'''

高效版

'''

class Solution(object):
  def productExceptSelf(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    n = len(nums)
    left = [1] * n
    for i in range(n-1):
     left[i+1] = nums[i]*left[i]
    right = [1] * n
    for j in range(n-1, 0, -1):
      right[j-1] = nums[j] * right[j]
    result = []
    for v1, v2 in zip(left, right):
      result.append(v1*v2)
    return result

'''

容易理解版

'''

class Solution(object):
  def productExceptSelf(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    n = len(nums)
    left = [0] * n
    left[0] = 1
    for i in range(1, n):
      left[i] = left[i-1] * nums[i-1]
    right = [0] * n
    _nums = nums[::-1]
    right[0] = 1
    for i in range(1, n):
      right[i] = right[i-1] * _nums[i-1]
    right = right[::-1]
    result = []
    for i in range(n):
      result.append(left[i] * right[i])
    return result