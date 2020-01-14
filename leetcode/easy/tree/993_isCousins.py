# coding=utf-8

# https://leetcode-cn.com/problems/cousins-in-binary-tree/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def isCousins(self, root, x, y):
    """
    :type root: TreeNode
    :type x: int
    :type y: int
    :rtype: bool
    """
    l = []
    l.append((0, None, root))
    x_node = None
    y_node = None
    while l:
      level, father, node = l[0]
      del l[0]
      if node.val == x:
        x_node = (level, father)
      if node.val == y:
        y_node = (level, father)
      if node.left:
        l.append((level+1, node, node.left))
      if node.right:
        l.append((level+1, node, node.right))
    if not x_node or not y_node:
      return False
    if x_node[0] == y_node[0] and x_node[1] != y_node[1]:
      return True
    return False
