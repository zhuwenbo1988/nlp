# coding=utf-8

'''

深度优先遍历

'''

# https://leetcode-cn.com/problems/path-sum-ii/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def pathSum(self, root, _sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: List[List[int]]
    """
    result = []
    if not root:
      return result
    def find(node, tmp):
      # 叶子节点
      if sum(tmp) == _sum and not node.left and not node.right:
        result.append(tmp)
        return
      if node.left:
        find(node.left, tmp + [node.left.val])
      if node.right:
        find(node.right, tmp + [node.right.val])
    find(root, [] + [root.val])
    return result