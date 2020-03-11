# coding=utf-8

'''

栈

'''

# https://leetcode-cn.com/problems/binary-tree-preorder-traversal/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def preorderTraversal(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
      return []
    l = []
    result = []
    l.append(root)
    while l:
      node = l[0]
      del l[0]
      result.append(node.val)
      if node.right:
        l.insert(0, node.right)
      if node.left:
        l.insert(0, node.left)
    return result
