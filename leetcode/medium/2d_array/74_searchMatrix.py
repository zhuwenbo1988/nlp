# coding=utf-8

# https://leetcode-cn.com/problems/search-a-2d-matrix/

class Solution(object):
  def searchMatrix(self, matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    """
    if not matrix:
      return False
    if not matrix[0]:
      return False
    m = len(matrix)
    n = len(matrix[0])
    idx = -1
    for i, values in enumerate(matrix):
      if target >= values[0] and target <= values[n-1]:
        idx = i
        break
    if idx < 0:
      return False
    values = matrix[idx]
    s = 0
    e = n-1
    while s <= e:
      mid = (s+e) // 2
      if target == values[mid]:
        return True
      if target > values[mid]:
        s = mid + 1
      else:
        e = mid - 1
    return False
