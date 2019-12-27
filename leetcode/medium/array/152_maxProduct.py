# coding=utf-8

'''

动态规划

我们只要记录前i的最小值, 和最大值, 那么 dp[i] = max(nums[i] * pre_max, nums[i] * pre_min, nums[i]), 这里0 不需要单独考虑, 因为当相乘不管最大值和最小值,都会置0

'''

# https://leetcode-cn.com/problems/maximum-product-subarray/

class Solution(object):
  def maxProduct(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    max_l = [0] * n
    min_l = [0] * n
    max_l[0] = nums[0]
    min_l[0] = nums[0]
    result = nums[0]

    for i in range(1, n):
      max_l[i] = max(max_l[i-1] * nums[i], min_l[i-1] * nums[i], nums[i])
      min_l[i] = min(max_l[i-1] * nums[i], min_l[i-1] * nums[i], nums[i])
      result = max(result, max_l[i])
    return result
