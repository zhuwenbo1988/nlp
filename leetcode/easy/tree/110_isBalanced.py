# coding=utf-8

'''

二叉树深度遍历

'''

# https://leetcode-cn.com/problems/balanced-binary-tree/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def isBalanced(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    self.is_balanced = True
    def dfs(node):
      if not node:
        return 0
      left_height = dfs(node.left)
      right_height = dfs(node.right)
      if abs(left_height - right_height) > 1:
          self.is_balanced = False
      return max(left_height, right_height) + 1
    dfs(root)
    return self.is_balanced