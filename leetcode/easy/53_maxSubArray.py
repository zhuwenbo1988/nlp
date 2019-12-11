# coding=utf-8

'''

动态规划

'''

# https://leetcode-cn.com/problems/maximum-subarray/

class Solution(object):
  def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    dp = [0] * len(nums)
    dp[0] = nums[0]
    for i in range(1, len(nums)):
      dp[i] = max(dp[i-1]+nums[i], nums[i])
    return max(dp)
