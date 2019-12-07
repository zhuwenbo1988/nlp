# coding=utf-8

# https://leetcode-cn.com/problems/validate-binary-search-tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def isValidBST(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    result = []
    is_valid = True
    l = []
    node = root
    while node or l:
      while node:
        l.insert(0, node)
        node = node.left
      node = l[0]
      del l[0]
      result.append(node.val)
      if len(result) > 1 and result[-2] >= result[-1]:
        is_valid = False
      node = node.right
    return is_valid
