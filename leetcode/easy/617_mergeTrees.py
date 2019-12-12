# coding=utf-8

'''

非递归,非常巧妙

'''

# https://leetcode-cn.com/problems/merge-two-binary-trees/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def mergeTrees(self, t1, t2):
    """
    :type t1: TreeNode
    :type t2: TreeNode
    :rtype: TreeNode
    """
    if not t1:
      return t2
    l = []
    l.append((t1, t2))
    while l:
      n1, n2 = l[0]
      del l[0]
      if not n1 or not n2:
        continue
      n1.val += n2.val
      if n1.left:
        l.append((n1.left, n2.left))
      else:
        # 巧妙
        n1.left = n2.left
      if n1.right:
        l.append((n1.right, n2.right))
      else:
        # 巧妙
        n1.right = n2.right
    return t1

