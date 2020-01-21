# coding=utf-8

# https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def minDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
      return 0
    l = []
    l.append((1, root))
    while l:
      level, node = l[0]
      del l[0]
      if not node.left and not node.right:
        return level
      if node.left:
        l.append((level+1, node.left))
      if node.right:
        l.append((level+1, node.right))
