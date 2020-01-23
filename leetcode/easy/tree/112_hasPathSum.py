# coding=utf-8

# https://leetcode-cn.com/problems/path-sum/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def hasPathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    if not root:
      return False
    l = []
    l.append((root, sum))
    while l:
      node, other = l[0]
      del l[0]
      if not node.left and not node.right and other-node.val == 0:
        return True
      if node.right:
        l.insert(0, (node.right, other-node.val))
      if node.left:
        l.insert(0, (node.left, other-node.val))
    return False
