# coding=utf-8

'''

并查集,还有不少优化方法

'''

# https://leetcode-cn.com/problems/friend-circles/

class Solution(object):
  def findCircleNum(self, M):
    """
    :type M: List[List[int]]
    :rtype: int
    """
    parent = [-1 for i in range(len(M))]

    def find(parent, i):
      while parent[i] != -1:
        i = parent[i]
      return i

    def union(i, j):
      root_i = find(parent, i)
      root_j = find(parent, j)
      if root_i != root_j:
        parent[root_i] = root_j

    def union_find(matrix):
      n = len(matrix)
      for i in range(n):
        for j in range(n):
          if i != j and matrix[i][j] == 1:
            union(i, j)

      count = 0
      for root in parent:
        if root == -1:
          count += 1
      return count

    return union_find(M)
