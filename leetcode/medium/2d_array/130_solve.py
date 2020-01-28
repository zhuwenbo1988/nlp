# coding=utf-8

# https://leetcode-cn.com/problems/surrounded-regions/

class Solution(object):
  def solve(self, board):
    """
    :type board: List[List[str]]
    :rtype: None Do not return anything, modify board in-place instead.
    """
    def dfs(board, row, col, nodes):
      if row < 0 or col < 0:
        return
      if row >= len(board) or col >= len(board[0]):
        return
      if board[row][col] == 'X':
        return
      board[row][col] = 'X'
      nodes.append((row, col))
      dfs(board, row-1, col, nodes)
      dfs(board, row+1, col, nodes)
      dfs(board, row, col-1, nodes)
      dfs(board, row, col+1, nodes)

    def in_boundary(data, row_nums, col_nums):
      for i, j in data:
        if i == 0 or j == 0:
          return True
        if i == row_nums or j == col_nums:
          return True
      return False

    if not board:
      return
    m = len(board)
    n = len(board[0])
    for i in range(m):
      for j in range(n):
        if board[i][j] == 'X':
          continue
        nodes = []
        dfs(board, i, j, nodes)
        if not in_boundary(nodes, m-1, n-1):
          continue
        for _i, _j in nodes:
          board[_i][_j] = 'O'
