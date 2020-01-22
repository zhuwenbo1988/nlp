# coding=utf-8

# https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Solution(object):
  def levelOrder(self, root):
    """
    :type root: Node
    :rtype: List[List[int]]
    """
    if not root:
      return []
    l = []
    l.append((1, root))
    levels = []
    while l:
      level, node = l[0]
      del l[0]
      if len(levels) < level:
        levels.append([])
      levels[level-1].append(node.val)
      if node.children:
        for child in node.children:
          l.append((level+1, child))
    return levels
