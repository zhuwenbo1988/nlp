# coding=utf-8

'''

将已经走过的地方置0，然后拐弯的时候判断一下是不是已经走过了，如果走过了就计算一下新的方向

好巧妙
'''

# https://leetcode-cn.com/problems/spiral-matrix/

class Solution(object):
  def spiralOrder(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    result = []
    if not matrix:
      return result
    i = 0
    j = 0
    di = 0
    dj = 1
    for _ in range(len(matrix) * len(matrix[0])):
      result.append(matrix[i][j])
      matrix[i][j] = 0
      m = (i + di) % len(matrix)
      n = (j + dj) % len(matrix[0])
      if matrix[m][n] == 0:
        tmp = di
        di = dj
        dj = -tmp
      i += di
      j += dj
    return result
