# coding=utf-8

'''

只能递归

'''

#  https://leetcode-cn.com/problems/delete-node-in-a-bst/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def deleteNode(self, root, key):
    """
    :type root: TreeNode
    :type key: int
    :rtype: TreeNode
    """
    if not root:
      return
    if root.val > key:
      # key在左子树，所以修正根结点的左子树
      root.left = self.deleteNode(root.left, key)
    elif root.val < key:
      root.right = self.deleteNode(root.right, key)
    else:
      # 左右子树其中一个为空
      if not root.left or not root.right:
        if root.left:
          root = root.left
        else:
          root = root.right
      # 难点
      else:
        # 取出右子树
        cur = root.right
        # 右子树的最左边的节点就是这颗右子树的根结点
        while cur.left:
          cur = cur.left
        # 改变根结点的值
        root.val = cur.val
        # 修正改变过值的根结点的右子树
        root.right = self.deleteNode(root.right, cur.val)
    return root
