# coding=utf-8

'''

回溯

'''

# https://leetcode-cn.com/problems/generate-parentheses/

class Solution(object):
  def generateParenthesis(self, n):
    """
    :type n: int
    :rtype: List[str]
    """
    result = []
    def find(curr_s, l, r):
      if l > n or r > n or r > l:
        return
      if l == n and r == n:
        result.append(''.join(curr_s))
        return
      # 加个左括号
      find(curr_s + ['('], l+1, r)
      # 加个右括号
      find(curr_s + [')'], l, r+1)
    find([], 0, 0)
    return result