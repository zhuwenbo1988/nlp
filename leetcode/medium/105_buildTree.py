# coding=utf-8

'''

非递归有难度,我是从别人的C++代码改的

'''

# https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def buildTree(self, preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    if not preorder and not inorder:
      return
    root = TreeNode(preorder[0])
    S = []
    S.append(root)
    i = 1
    j = 0
    while i < len(preorder):
      back = None
      curr = TreeNode(preorder[i])
      while S and S[0].val == inorder[j]:
        back = S[0]
        del S[0]
        j += 1
      if back:
        back.right = curr
      else:
        S[0].left = curr
      S.insert(0, curr)
      i += 1
    return root
