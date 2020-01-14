# coding=utf-8

'''

完全背包问题
01背包是每种只有一件，完全背包是每种无限件

'''

# https://leetcode-cn.com/problems/combination-sum-iv/

class Solution(object):
  def combinationSum4(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    dp = [0] * (target+1)
    dp[0] = 1
    for i in range(1, target+1):
      for v in nums:
        if v <= i:
          dp[i] += dp[i-v]
    return dp[-1]
