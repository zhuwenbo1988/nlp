# coding=utf-8

# https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def zigzagLevelOrder(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    if not root:
      return []
    l = []
    l.append((0, root))
    levels = [[]]
    while l:
      level, node = l[0]
      del l[0]
      if len(levels) <= level:
        levels.append([])
      if level % 2 == 0:
        levels[level].append(node.val)
      else:
        levels[level].insert(0, node.val)
      if node.left:
        l.append((level+1, node.left))
      if node.right:
        l.append((level+1, node.right))
    return levels
