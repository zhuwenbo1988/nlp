# coding=utf-8

# https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def kthSmallest(self, root, k):
    """
    :type root: TreeNode
    :type k: int
    :rtype: int
    """
    stack = []
    l = []
    while root or stack:
      while root:
        stack.insert(0, root)
        root = root.left
      root = stack[0]
      del stack[0]
      l.append(root.val)
      if len(l) == k:
        break
      root = root.right
    return l[k-1]
