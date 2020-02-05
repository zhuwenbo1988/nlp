# coding=utf-8

'''

原地转换等同于用right表示next

链表的结果是先序遍历的,但是本题的方法是在中序遍历的基础上进行的

'''
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
      # 取出中序遍历的节点
      node = l[0]
      del l[0]
      # 左子树放到右子树，左子树置空
      tmp = node.right
      node.right = node.left
      node.left = None
      # 将原来的右子树放到原来的左子树(现在是当前节点的右子树)的最后一个节点
      while node.right:
        node = node.right
      node.right = tmp
      # 中序遍历最后要处理右节点
      root = tmp