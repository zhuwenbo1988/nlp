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
      # 1.打印
      result.append(matrix[i][j])
      # 2.置0
      matrix[i][j] = 0
      # 3.下一步是否已经走过,如果已经走过,则交换di和dj
      next_i = (i + di) % len(matrix)
      next_j = (j + dj) % len(matrix[0])
      if matrix[next_i][next_j] == 0:
        tmp = di
        di = dj
        dj = -tmp
      # 4.确定下一步
      i += di
      j += dj
    return result
