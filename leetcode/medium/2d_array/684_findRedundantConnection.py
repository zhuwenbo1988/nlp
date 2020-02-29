# coding=utf-8

'''

并查集

'''

# https://leetcode-cn.com/problems/redundant-connection/

class Solution(object):
  def findRedundantConnection(self, edges):
    """
    :type edges: List[List[int]]
    :rtype: List[int]
    """
    parent = [-1 for i in range(2000)]
    
    def find(parent, i):
      while parent[i] != -1:
        i = parent[i]
      return i

    def union(i, j):
      root_i = find(parent, i)
      root_j = find(parent, j)
      if root_i == root_j:
        return False
      parent[root_i] = root_j
      return True

    edge = []
    for i, j in edges:
      if not union(i, j):
        edge.append((i, j))
        
    return edge[-1]
