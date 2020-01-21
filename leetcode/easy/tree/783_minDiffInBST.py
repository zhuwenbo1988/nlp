# coding=utf-8

# https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def minDiffInBST(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    nodes = []
    l = []
    while l or root:
      while root:
        l.insert(0, root)
        root = root.left
      root = l[0]
      del l[0]
      nodes.append(root.val)
      root = root.right
    min_gap = 10000
    print nodes
    for i in range(1, len(nodes)):
      gap = abs(nodes[i] - nodes[i-1])
      if gap < min_gap:
        min_gap = gap
    return min_gap
