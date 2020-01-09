# coding=utf-8

# https://leetcode-cn.com/problems/valid-parentheses/

class Solution(object):
  def isValid(self, s):
    """
    :type s: str
    :rtype: bool
    """
    if not s:
      return True
    def is_pair(s1, s2):
      if s1 == '(' and s2 == ')':
        return True
      if s1 == '{' and s2 == '}':
        return True
      if s1 == '[' and s2 == ']':
        return True
      return False
    l = []
    l.append(s[0])
    for i in range(1, len(s)):
      if l:
        if is_pair(l[0], s[i]):
          del l[0]
        else:
          l.insert(0, s[i])
      else:
        l.insert(0, s[i])
    if l:
      return False
    return True
