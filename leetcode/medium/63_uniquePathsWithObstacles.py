# coding=utf-8

# https://leetcode-cn.com/problems/unique-paths-ii/

class Solution(object):
  def uniquePathsWithObstacles(self, obstacleGrid):
    """
    :type obstacleGrid: List[List[int]]
    :rtype: int
    """
    m = len(obstacleGrid)
    n = len(obstacleGrid[0])
    result = [[0]*n for i in range(m)]
    for i in range(m):
      for j in range(n):
        if obstacleGrid[i][j] == 1:
          result[i][j] = 0
          continue
        if i == 0 and j == 0:
          result[i][j] = 1 
          continue
        result[i][j] = result[i][j-1] + result[i-1][j]
    return result[m-1][n-1]
