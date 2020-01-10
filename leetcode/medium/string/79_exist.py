# coding=utf-8

# https://leetcode-cn.com/problems/word-search/

class Solution(object):
  def exist(self, board, word):
    """
    :type board: List[List[str]]
    :type word: str
    :rtype: bool
    """
    self.path = []
    # 从矩阵的每个点开始寻找
    for i, row in enumerate(board):
      for j, _ in enumerate(row):
        if self.find(board, word, i, j):
          return True
    return False
  
  def find(self, board, word, i, j):
    '''
    i,j是起点
    '''
    # 超出边界
    if i < 0 or j < 0:
      return False
    # 超出边界
    if i >= len(board) or j >= len(board[0]):
      return False
    # 走过的点就不能走了
    if (i,j) in self.path:
      return False
    # 走不通的点就不能走了
    if word[0] != board[i][j]:
      return False

    # 当前的点是合格的
    self.path.append((i, j))

    # 判断后面的词
    word = word[1:]

    # 全部匹配
    if not word:
      return True

    # 从当前点的前后左右再次出发
    if self.find(board,word,i-1,j) or self.find(board,word,i,j-1) or self.find(board,word,i+1,j) or self.find(board,word,i,j+1):
      return True

    # 前后左右都不行,从路径中去掉这个点
    self.path.pop()
    return False
