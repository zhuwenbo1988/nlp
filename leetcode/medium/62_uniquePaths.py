# coding=utf-8

'''

经典动态规划问题

'''

# https://leetcode-cn.com/problems/unique-paths/

class Solution(object):
  def uniquePaths(self, m, n):
    """
    :type m: int
    :type n: int
    :rtype: int
    """
    matrix = [[0 for j in range(n)] for i in range(m)]
    for i in range(m):
      for j in range(n):
        if i == 0 and j == 0:
          matrix[i][j] = 1
        else:
          matrix[i][j] = matrix[i][j-1] + matrix[i-1][j]
    return matrix[m-1][n-1]
