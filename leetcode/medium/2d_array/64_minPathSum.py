# coding=utf-8

'''

经典动态规划

'''

# https://leetcode-cn.com/problems/minimum-path-sum

class Solution(object):
  def minPathSum(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    m = len(grid)
    n = len(grid[0])
    dp = [[0] * n for i in range(m)]
    for i in range(m):
      for j in range(n):
        if i == 0 and j == 0:
          dp[i][j] = grid[i][j]
        else:
          if i-1 < 0:
            dp[i][j] = grid[i][j] + dp[i][j-1]
          elif j-1 < 0:
            dp[i][j] = grid[i][j] + dp[i-1][j]
          else:
            dp[i][j] = min(grid[i][j] + dp[i-1][j], grid[i][j] + dp[i][j-1])
    return dp[m-1][n-1]
