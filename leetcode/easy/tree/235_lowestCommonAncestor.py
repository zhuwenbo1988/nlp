# coding=utf-8

'''

利用二叉搜索树的属性

'''

# https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def lowestCommonAncestor(self, root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    l = []
    l.append(root)
    while l:
      node = l[0]
      del l[0]
      if p.val > node.val and q.val > node.val:
        l.append(node.right)
        continue
      if p.val < node.val and q.val < node.val:
        l.append(node.left)
        continue
      if p.val == node.val or q.val == node.val:
        return node
      if p.val < node.val and q.val > node.val:
        return node
      if p.val > node.val and q.val < node.val:
        return node
