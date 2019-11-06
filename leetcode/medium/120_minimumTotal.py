# coding=utf-8

'''

动态规划，关键是i和j的使用

'''

# https://leetcode-cn.com/problems/triangle/

class Solution(object):
  def minimumTotal(self, triangle):
    for i in range(len(triangle) - 1, 0, -1):
      for j in range(i):
        triangle[i - 1][j] += min(triangle[i][j], triangle[i][j + 1])
    return triangle[0][0]
