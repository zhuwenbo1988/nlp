# coding=utf-8

# https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def flatten(self, root):
    """
    :type root: TreeNode
    :rtype: None Do not return anything, modify root in-place instead.
    """
    l = []
    while l or root:
      while root:
        l.insert(0, root)
        root = root.left

      if l:
        node = l[0]
        del l[0]
        tmp = node.right
        #左子树放到右子树，左子树置空
        node.right = node.left
        node.left = None

        # 右子树最后一个节点
        while node.right:
          node = node.right
        node.right = tmp
        # 从右子树开始
        root = tmp
