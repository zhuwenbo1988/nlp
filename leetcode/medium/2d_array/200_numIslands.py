# coding=utf-8

'''

深度优先遍历

线性扫描整个二维网格，如果一个结点包含1，则以其为根结点启动深度优先搜索。在深度优先搜索过程中，每个访问过的结点被标记为0。计数启动深度优先搜索的根结点的数量，即为岛屿的数量。

时  间复杂度 : O(M×N)，其中 M 和 N 分别为行数和列数。
空间复杂度 : 最坏情况下为O(M×N)，此时整个网格均为陆地，深度优先搜索的深度达到M×N。

'''

# https://leetcode-cn.com/problems/number-of-islands/

class Solution(object):
  def numIslands(self, grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """
    def dfs(grid, row, col):
      if row < 0 or col < 0:
        return
      if row >= len(grid) or col >= len(grid[0]):
        return
      if grid[row][col] == '0':
        return

      grid[row][col] = '0'
    
      dfs(grid, row-1, col)
      dfs(grid, row+1, col)
      dfs(grid, row, col-1)
      dfs(grid, row, col+1)
    if not grid:
      return 0
    m = len(grid)
    n = len(grid[0])
    result = 0
    for i in range(m):
      for j in range(n):
        if grid[i][j] == '0':
          continue
        result += 1
        dfs(grid, i, j)
    return result
