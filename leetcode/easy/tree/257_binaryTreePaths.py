# coding=utf-8

# https://leetcode-cn.com/problems/binary-tree-paths/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def binaryTreePaths(self, root):
    """
    :type root: TreeNode
    :rtype: List[str]
    """
    if not root:
      return []
    l = []
    l.append((root, str(root.val)))
    result = []
    while l:
      node, path = l[0]
      del l[0]
      if not node.left and not node.right:
        result.append(path)
      if node.left:
        l.append((node.left, '{}->{}'.format(path, node.left.val)))
      if node.right:
        l.append((node.right, '{}->{}'.format(path, node.right.val)))
    return result
