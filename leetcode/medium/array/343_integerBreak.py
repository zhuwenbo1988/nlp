# coding=utf-8

'''

令dp[i]表示整数i对应的最大乘积，那么dp[i]的值应是dp[j]*(i-j),j属于[1,i-1]的最大值，同时注意dp[i]对应的值是经过拆分了的，所以还应判断两个数拆分的情况，即j*(i-j)的值，取最大即可。

'''

# https://leetcode-cn.com/problems/integer-break/

class Solution(object):
  def integerBreak(self, n):
    """
    :type n: int
    :rtype: int
    """
    dp = [0] * (n+1)
    dp[1] = 1
    for i in range(2, n+1):
      for j in range(1, i):
        dp[i] = max([dp[i], dp[j]*(i-j), j*(i-j)])
    return dp[n]
