# coding=utf-8

'''

动态规划

当前台阶可以从上一个和上上一个台阶到达

'''

# https://leetcode-cn.com/problems/climbing-stairs/

class Solution(object):
  def climbStairs(self, n):
    """
    :type n: int
    :rtype: int
    """
    dp = [0] * (n+1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n+1):
      dp[i] = dp[i-1] + dp[i-2]
    return dp[-1]
