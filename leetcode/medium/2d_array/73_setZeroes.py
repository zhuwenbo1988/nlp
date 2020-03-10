# coding=utf-8

# https://leetcode-cn.com/problems/set-matrix-zeroes/

class Solution(object):
  def setZeroes(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: None Do not return anything, modify matrix in-place instead.
    """
    m = len(matrix)
    n = len(matrix[0])
    modify = -10000
    for i in range(m):
      for j in range(n):
        if matrix[i][j] != 0:
          continue
        for _i in range(m):
          if matrix[_i][j] == 0:
            continue
          matrix[_i][j] = modify
        for _j in range(n):
          if matrix[i][_j] == 0:
            continue
          matrix[i][_j] = modify

    for i in range(m):
      for j in range(n):
        if matrix[i][j] == modify:
          matrix[i][j] = 0