# coding=utf-8

'''

其实就是求ｓ和ｓ的逆序的最长公共子序列的。转化成最长公共子序列问题就迎刃而解了。

'''

# https://leetcode-cn.com/problems/longest-palindromic-subsequence/

class Solution(object):
  def longestPalindromeSubseq(self, s):
    """
    :type s: str
    :rtype: int
    """
    s1 = s
    s2 = s[::-1]
    m = len(s1)
    n = m
    dp = [[0]*(m+1) for i in range(n+1)]
    for i in range(1, n+1):
      for j in range(1, m+1):
        if s1[i-1] == s2[j-1]:
          dp[i][j] = 1 + dp[i-1][j-1]
        else:
          dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
