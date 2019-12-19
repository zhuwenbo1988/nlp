# coding=utf-8

'''

用值进行二分查找

'''

# https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/

class Solution(object):
  def kthSmallest(self, matrix, k):
    """
    :type matrix: List[List[int]]
    :type k: int
    :rtype: int
    """
    m = len(matrix)
    n = len(matrix)
    # 最小值
    left = matrix[0][0]
    # 最大值
    right = matrix[-1][-1]
    while left < right:
      # 中间值
      mid_val = (left+right) // 2
      count = self.find(matrix, mid_val, m, n)
      if count < k:
        left = mid_val + 1
      else:
        right = mid_val
    return right

  def find(self, matrix, mid_v, row, col):
    count = 0
    # i从最大开始,递减;j从最小开始,递增
    i = row - 1
    j = 0
    while i >= 0 and j < col:
      if matrix[i][j] <= mid_v:
        count += (i+1)
        # 增加j
        j += 1
      else:
        # 减小i
        i += -1
    return count
