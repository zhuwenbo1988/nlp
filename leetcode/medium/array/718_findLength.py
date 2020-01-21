# coding=utf-8

'''

参考最长公共子串(要求连续)

'''

# https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/

class Solution(object):
  def findLength(self, A, B):
    """
    :type A: List[int]
    :type B: List[int]
    :rtype: int
    """
    m = len(A)
    n = len(B)
    dp = [[0] * (n+1) for _ in range(m+1)]
    max_len = 0
    for i in range(1, m+1):
      for j in range(1, n+1):
        if A[i-1] == B[j-1]:
          dp[i][j] = dp[i-1][j-1] + 1
          max_len = max(max_len, dp[i][j])
    return max_len
