# coding=utf-8

'''

贪心
该算法通用且简单：遍历数组并在每个步骤中更新：
当前元素
当前元素位置的最大和
迄今为止的最大和

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
