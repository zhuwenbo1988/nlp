# coding=utf-8

'''

栈

'''

# https://leetcode-cn.com/problems/binary-tree-inorder-traversal

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def inorderTraversal(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    result = []
    l = []
    node = root
    while node or l:
      while node:
        l.insert(0, node)
        node = node.left
      node = l[0]
      del l[0]
      result.append(node.val)
      node = node.right
    return result
