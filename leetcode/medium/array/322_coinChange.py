# coding=utf-8

'''

背包问题

'''

# https://leetcode-cn.com/problems/coin-change/

import sys

class Solution(object):
  def coinChange(self, coins, amount):
    """
    :type coins: List[int]
    :type amount: int
    :rtype: int
    """
    # dp[i] 表示金额为i需要最少的硬币个数
    dp = [sys.maxint] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount+1):
      for c in coins:
         # 没有超过背包的容量
         if c <= i:
           # 原来的个数 or 个数+1，看哪个小
           dp[i] = min(dp[i], dp[i-c]+1)
    if dp[-1] != sys.maxint:
      return dp[-1]
    return -1
