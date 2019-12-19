# coding=utf-8

# https://leetcode-cn.com/problems/find-bottom-left-tree-value/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import defaultdict

class Solution(object):
  def findBottomLeftValue(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    l = []
    max_level = 0
    result = defaultdict(list)
    l.append((0, root))
    while l:
      level, root = l[0]
      del l[0]
      if level > max_level:
        max_level = level
      result[level].append(root.val)
      if root.left:
        l.append((level+1, root.left))
      if root.right:
        l.append((level+1, root.right))
    return result[max_level][0]
