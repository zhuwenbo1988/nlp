# coding=utf-8

# https://leetcode-cn.com/problems/min-stack/

class MinStack(object):

  def __init__(self):
    """
    initialize your data structure here.
    """
    self.stack = []
    self.help_stack = []


  def push(self, x):
    """
    :type x: int
    :rtype: None
    """
    self.stack.insert(0, x)
    if not self.help_stack or x <= self.help_stack[0]:
      self.help_stack.insert(0, x)


  def pop(self):
    """
    :rtype: None
    """
    if not self.stack:
      return
    x = self.stack[0]
    del self.stack[0]
    if x == self.help_stack[0]:
      del self.help_stack[0]
    return x

  def top(self):
    """
    :rtype: int
    """
    if not self.stack:
      return
    return self.stack[0]

  def getMin(self):
    """
    :rtype: int
    """
    if not self.help_stack:
      return
    return self.help_stack[0]