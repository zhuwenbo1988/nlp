# coding=utf-8

'''

python 由于每个节点只有唯一一个父节点，我想到了使用字典存储各个节点的父节点，字典中预置根节点的父节点为None。字典建立完成后，二叉树就可以看成一个所有节点都将指向根节点的链表了。于是在二叉树中寻找两个节点的最小公共节点就相当于，在一个链表中寻找他们相遇的节点

'''

# https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/

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
    path = {root:None}
    def bfs(node):
      if node:
        if node.left:
          path[node.left] = node
        if node.right:
          path[node.right] = node
        bfs(node.left)
        bfs(node.right)
    bfs(root)
    # 寻找公共节点,总会碰到的
    l1 = p
    l2 = q
    while l1 != l2:
      l1 = path[l1]
      if not l1:
        l1 = q
      l2 = path[l2]
      if not l2:
        l2 = p
    return l1
