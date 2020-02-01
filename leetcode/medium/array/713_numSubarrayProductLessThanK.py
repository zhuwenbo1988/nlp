# coding=utf-8

'''

双指针

难以想到，背吧

'''

# https://leetcode-cn.com/problems/subarray-product-less-than-k/

class Solution(object):
  def numSubarrayProductLessThanK(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    if k <= 1:
      return 0
    j = 0
    n = len(nums)
    result = 0
    curr_p = 1
    for i in range(n):
      curr_p *= nums[i]
      while curr_p >= k:
        curr_p /= nums[j]
        j += 1
      # 关键
      result += i-j+1
    return result
