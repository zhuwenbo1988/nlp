# coding=utf-8

# https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def sumNumbers(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
      return 0
    result = []
    d = {}
    d[root] = None
    l = []
    l.append(root)
    while l:
      node = l[0]
      del l[0]
      if not node.left and not node.right:
        s = []
        father = d[node]
        s.insert(0, str(node.val))
        while father:
          s.insert(0, str(father.val))
          father = d[father]
        result.append(int(''.join(s)))
        continue
      if node.right:
        l.insert(0, node.right)
        d[node.right] = node
      if node.left:
        l.insert(0, node.left)
        d[node.left] = node
    return sum(result)
