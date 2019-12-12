# coding=utf-8

# https://leetcode-cn.com/problems/invert-binary-tree/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def invertTree(self, root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    if not root:
      return root
    l = []
    l.append(root)
    while l:
      node = l[0]
      del l[0]
      tmp = node.left
      node.left = node.right
      node.right = tmp
      if node.left:
        l.append(node.left)
      if node.right:
        l.append(node.right)
    return root
