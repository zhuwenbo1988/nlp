# coding=utf-8

'''

动态规划

我们只要记录前i(包括i，因为连续)的最小值, 和最大值, 那么 dp[i] = max(nums[i] * pre_max, nums[i] * pre_min, nums[i]), 这里0 不需要单独考虑, 因为当相乘不管最大值和最小值,都会置0

'''

# https://leetcode-cn.com/problems/maximum-product-subarray/

class Solution(object):
  def maxProduct(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    max_dp = [0] * n
    min_dp = [0] * n
    max_dp[0] = nums[0]
    min_dp[0] = nums[0]

    for i in range(1, n):
      max_dp[i] = max(max_dp[i-1] * nums[i], min_dp[i-1] * nums[i], nums[i])
      min_dp[i] = min(max_dp[i-1] * nums[i], min_dp[i-1] * nums[i], nums[i])
    return max(max_dp)
