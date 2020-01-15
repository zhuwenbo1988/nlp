# coding=utf-8

'''

要利用完全二叉树的性质

'''

# https://leetcode-cn.com/problems/count-complete-tree-nodes/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def countNodes(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
      return 0
    left_node = root
    left_h = 0
    right_node = root
    right_h = 0
    while left_node:
      left_node = left_node.left
      left_h += 1
    while right_node:
      right_node = right_node.right
      right_h += 1
    if left_h == right_h:
      return 2**(left_h)-1
    return 1 + self.countNodes(root.left) + self.countNodes(root.right)
