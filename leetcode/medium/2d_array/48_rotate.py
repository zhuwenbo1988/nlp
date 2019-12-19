# coding=utf-8

'''

这道题是反对角线交换,很难,所以只能拆分为简单的先上下交换再对角交换

'''

# https://leetcode-cn.com/problems/rotate-image/

class Solution(object):
  def rotate(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: None Do not return anything, modify matrix in-place instead.
    """
    n = len(matrix)
    for i in range(n // 2):
      tmp = matrix[n-i-1]
      matrix[n-i-1] = matrix[i]
      matrix[i] = tmp
    for i in range(n):
      for j in range(i):
        tmp = matrix[i][j]
        matrix[i][j] = matrix[j][i]
        matrix[j][i] = tmp
    return matrix
