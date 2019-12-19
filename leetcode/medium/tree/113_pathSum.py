# coding=utf-8

# https://leetcode-cn.com/problems/path-sum-ii/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def pathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: List[List[int]]
    """
    result = []
    if not root:
      return result
    def find(node, curr_sum, tmp):
      if curr_sum == sum and not node.left and not node.right:
        result.append(tmp)
        return
      if node.left:
        find(node.left, curr_sum + node.left.val, tmp + [node.left.val])
      if node.right:
        find(node.right, curr_sum + node.right.val, tmp + [node.right.val])
    find(root, root.val, [] + [root.val])
    return result
