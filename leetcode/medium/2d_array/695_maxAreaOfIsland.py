# coding=utf-8

# https://leetcode-cn.com/problems/max-area-of-island/

class Solution(object):
  def maxAreaOfIsland(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    def dfs(grid, r, c, area):
      if r < 0 or c < 0:
        return
      if r >= len(grid) or c >= len(grid[0]):
        return
      if grid[r][c] == 0:
        return

      grid[r][c] = 0
      area.append((r, c))
      dfs(grid, r-1, c, area)
      dfs(grid, r+1, c, area)
      dfs(grid, r, c-1, area)
      dfs(grid, r, c+1, area)
    
    m = len(grid)
    n = len(grid[0])
    max_area = 0
    for i in range(m):
      for j in range(n):
        if grid[i][j] == 0:
          continue
        area = []
        dfs(grid, i, j, area)
        if max_area < len(area):
          max_area = len(area)
    return max_area
