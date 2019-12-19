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
    for idx, i in enumerate(board):
      for jdx, j in enumerate(i):
        if self.find(board, word, idx, jdx):
          return True
    return False
  
  def find(self, board, word, i, j):
    if i < 0 or j < 0:
      return False
    if i >= len(board) or j >= len(board[0]):
      return False
    if (i,j) in self.path:
      return False
    if word[0] != board[i][j]:
      return False
    
    word = word[1:]
    self.path.append((i,j))   
  
    if not word:
      return True 
      
    if self.find(board,word,i-1,j) or self.find(board,word,i,j-1) or self.find(board,word,i+1,j) or self.find(board,word,i,j+1):
      return True
    
    self.path.pop()
    return False
