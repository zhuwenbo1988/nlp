# coding=utf-8

'''

参考题目300，最长上升子序列

'''

# https://leetcode-cn.com/problems/maximum-length-of-pair-chain/

class Solution(object):
  def findLongestChain(self, pairs):
    """
    :type pairs: List[List[int]]
    :rtype: int
    """
    pairs.sort()
    n = len(pairs)
    dp = [1] * n
    for i in range(1, n):
       for j in range(i):
         if pairs[i][0] > pairs[j][1]:
           dp[i] = max(dp[i], dp[j]+1)
    return max(dp)
