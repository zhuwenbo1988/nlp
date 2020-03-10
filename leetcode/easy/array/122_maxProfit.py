# coding=utf-8

'''

关键是我们需要考虑到紧跟谷的每一个峰值以最大化利润。如果我们试图跳过其中一个峰值来获取更多利润，那么我们最终将失去其中一笔交易中获得的利润，从而导致总利润的降低。

'''

# https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/

class Solution(object):
  def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    max_profit = 0
    i = 0
    while i < len(prices)-1:
      while i < len(prices)-1 and prices[i] >= prices[i+1]:
        i += 1
      bogu = prices[i]
      while i < len(prices)-1 and prices[i] <= prices[i+1]:
        i += 1
      bofeng = prices[i]
      max_profit += bofeng-bogu
    return max_profit