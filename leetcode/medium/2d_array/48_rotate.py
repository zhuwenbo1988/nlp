# coding=utf-8

'''

先转置,再水平镜像翻转,二者都是简单的操作

'''

# https://leetcode-cn.com/problems/rotate-image/

class Solution(object):
  def rotate(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: None Do not return anything, modify matrix in-place instead.
    """
    # 转置
    n = len(matrix)
    for i in range(n):
      for j in range(i):
        tmp = matrix[i][j]
        matrix[i][j] = matrix[j][i]
        matrix[j][i] = tmp
    # 水平镜像翻转
    s = 0
    e = n
    while s < e:
      for i in range(n):
        tmp = matrix[i][s]
        matrix[i][s] = matrix[i][e]
        matrix[i][e] = tmp
      s += 1
      e += -1
    return matrix