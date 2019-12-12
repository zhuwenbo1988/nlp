# coding=utf-8

# https://leetcode-cn.com/problems/diameter-of-binary-tree/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def diameterOfBinaryTree(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    self.result = 1
    def depth(node):
      if not node:
        return 0
      left_depth = depth(node.left)
      right_depth = depth(node.right)
      self.result = max(self.result, left_depth+right_depth+1)
      return max(left_depth, right_depth) + 1
    depth(root)
    return self.result - 1
