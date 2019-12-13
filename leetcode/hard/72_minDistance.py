# coding=utf-8

'''

完全不会

'''

# https://leetcode-cn.com/problems/edit-distance/

class Solution(object):
  def minDistance(self, word1, word2):
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    m = len(word1)
    n = len(word2)
    
    dp = [[0]*(n+1) for _ in range(m+1)]
    
    # 把word1的所有字符都改成word2的第一个字符所需的次数
    for i in range(m+1):
      dp[i][0] = i
    
    for j in range(n+1):
      dp[0][j] = j

    for i in range(1, m+1):
      for j in range(1, n+1):
        # 当前相等
        if word1[i-1] == word2[j-1]:
          # 不用改
          dp[i][j] = dp[i-1][j-1]
        else:
          # 其中，dp[i-1][j-1] 表示替换操作，dp[i-1][j] 表示删除操作，dp[i][j-1] 表示插入操作。
          dp[i][j] = 1 + min(dp[i-1][j-1], min(dp[i][j-1], dp[i-1][j]))
    return dp[m][n]

