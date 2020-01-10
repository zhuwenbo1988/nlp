# coding=utf-8

'''

只能递归

先找要删除的节点
找到后,分四种情况处理
1.叶子节点
直接删除
2.只有左子树
用左子树代替当前节点
3.只有右子树
用右子树代替当前节点
4.左右子树都有
先找到右子树的左下方节点
将当前节点置为左下方节点
然后删除右子树中的左下方节点

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
      # 叶子节点
      if not root.left and not root.right:
        root = None
      elif root.left and not root.right: # 只有左子树
        root = root.left
      elif not root.left and root.right: # 只有右子树
        root = root.right
      # 难点: 左右子树都不空
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
