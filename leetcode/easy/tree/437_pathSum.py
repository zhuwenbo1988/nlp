# coding=utf-8

'''

超时

'''

# https://leetcode-cn.com/problems/path-sum-iii/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def pathSum(self, root, k):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: int
    """
    if not root:
      return 0
    def count(nums):
      total = 0
      for i in range(len(nums)):
        if sum(nums[i:]) == k:
          total += 1
      return total
    result = 0
    l = []
    l.append((root, [root.val]))
    while l:
      node, path = l[0]
      del l[0]
      result += count(path)
      if node.left:
        l.append((node.left, path + [node.left.val]))
      if node.right:
        l.append((node.right, path + [node.right.val]))
    return result
