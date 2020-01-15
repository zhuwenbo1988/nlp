# coding=utf-8

'''

很巧妙

'''

# https://leetcode-cn.com/problems/check-completeness-of-a-binary-tre

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def isCompleteTree(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    l = []
    l.append((0, 1, root))
    levels = []
    while l:
      level, posi, node = l[0]
      del l[0]
      levels.append((level, posi))
      if node.left:
        l.append((level+1, posi*2, node.left))
      if node.right:
        l.append((level+1, posi*2+1, node.right))
    return levels[-1][1] == len(levels)
