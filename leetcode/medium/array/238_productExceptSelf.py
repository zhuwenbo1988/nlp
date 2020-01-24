# coding=utf-8

# https://leetcode-cn.com/problems/product-of-array-except-self/

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
