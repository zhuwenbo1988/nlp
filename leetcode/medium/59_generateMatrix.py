# coding=utf-8

# https://leetcode-cn.com/problems/spiral-matrix-ii

class Solution(object):
  def generateMatrix(self, n):
    """
    :type n: int
    :rtype: List[List[int]]
    """
    nums = range(1, n*n + 1)
    M = [[1]*n for i in range(n)]
    result = [[0]*n for i in range(n)]
    i = 0
    j = 0
    di = 0
    dj = 1
    for v in nums:
      result[i][j] = v
      M[i][j] = 0
      m1 = (i + di) % n
      n1 = (j + dj) % n
      if M[m1][n1] == 0:
        tmp = di
        di = dj
        dj = -tmp
      i = i + di
      j = j + dj
    return result
