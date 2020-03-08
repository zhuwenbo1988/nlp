# coding=utf-8 

'''

动态规划

'''

# https://leetcode-cn.com/problems/palindromic-substrings/

class Solution(object):
  def countSubstrings(self, s):
    """
    :type s: str
    :rtype: int
    """
    result = 0
    n = len(s)
    
    # dp[i][j] 表示以j开头以i结尾的字符是否为回文子串
    dp = [[0]*n for i in range(n)]
    # 外层循环要倒着写，内层循环要正着写
    for i in range(n-1, -1, -1):
      for j in range(i, n):
        if s[i] == s[j]:
          # i和j中间最多只能有一个字母 or
          if j-i <= 2 or dp[i+1][j-1] == 1:
            dp[i][j] = 1
            result += 1
    return result
