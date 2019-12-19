# coding=utf-8

# https://leetcode-cn.com/problems/search-a-2d-matrix-ii/

class Solution(object):
  def searchMatrix(self, matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    """
    m = len(matrix)
    if m == 0:
      return False
    n = len(matrix[0])
    if n == 0:
      return False
    i = m-1
    j = 0
    while i >= 0 and j < n:
      v = matrix[i][j]
      if v == target:
        return True
      if v > target:
        i += -1
      else:
        j += 1
    return False
