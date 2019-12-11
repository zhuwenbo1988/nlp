# coding=utf-8

# https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/

class Solution(object):
  def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if not prices:
      return 0
    # 最大利润
    max_p = 0
    # 最小价格
    min_p = prices[0]
    for p in prices[1:]:
      max_p = max(max_p, p - min_p)
      min_p = min(min_p, p)
    return max_p
