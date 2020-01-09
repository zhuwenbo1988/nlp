# coding=utf-8

'''

想法
任意一条路径可以被写成两个 箭头（不同方向），每个箭头代表一条从某些点向下遍历到孩子节点的路径。
假设我们知道对于每个节点最长箭头距离分别为 L, RL,R，那么最优路径经过 L + R + 1 个节点。

算法
按照常用方法计算一个节点的深度：max(depth of node.left, depth of node.right) + 1。在计算的同时，经过这个节点的路径长度为 1 + (depth of node.left) + (depth of node.right) 。搜索每个节点并记录这些路径经过的点数最大值，期望长度是结果 - 1。

'''

# https://leetcode-cn.com/problems/diameter-of-binary-tree/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def diameterOfBinaryTree(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    self.result = 1
    def depth(node):
      if not node:
        return 0
      left_depth = depth(node.left)
      right_depth = depth(node.right)
      self.result = max(self.result, left_depth+right_depth+1)
      return max(left_depth, right_depth) + 1
    depth(root)
    return self.result - 1
