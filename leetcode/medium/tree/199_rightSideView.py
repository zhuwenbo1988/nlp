# coding=utf-8

# https://leetcode-cn.com/problems/binary-tree-right-side-view/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import defaultdict

class Solution(object):
  def rightSideView(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
      return []
    result = defaultdict(list)
    i = 1
    l = []
    l.append((i, root))
    while l:
      i, node = l[0]
      del l[0]
      result[i].append(node.val)
      if node.left:
        l.append((i+1, node.left))
      if node.right:
        l.append((i+1, node.right))
    res = []
    for i, nodes in sorted(result.iteritems(), key=lambda x: x[0], reverse=False):
      res.append(nodes[-1])
    return res
