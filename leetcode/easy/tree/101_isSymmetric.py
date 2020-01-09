# coding=utf-8

# https://leetcode-cn.com/problems/symmetric-tree/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import defaultdict

class Solution(object):
  def isSymmetric(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    if not root:
      return True
    # 1.层次遍历
    result = defaultdict(list)
    l = []
    l.append((1, root))
    while l:
      level, node = l[0]
      del l[0]
      if not node:
        result[level].append(None)
        continue
      else:
        result[level].append(node.val)
      # 只要node不是None,即使左右节点是None,也要入栈
      l.append((level+1, node.left))
      l.append((level+1, node.right))
    # 判断每一层是不是回文
    for level in result:
      nums = result[level]
      if not nums:
        continue
      if nums != nums[::-1]:
        return False
    return True
