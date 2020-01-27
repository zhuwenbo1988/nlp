# coding=utf-8

'''

完全二叉树有一个非常重要的性质:
若节点位置位k, 则lchild位置位2k, rchild位置位2k+1; (从1开始计算)

'''

# https://leetcode-cn.com/problems/complete-binary-tree-inserter/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class CBTInserter(object):

  def __init__(self, root):
    """
    :type root: TreeNode
    """
    self.root = root
    self.node_index_dict, self.curr_max_index = self._build(root)

  def insert(self, v):
    """
    :type v: int
    :rtype: int
    """
    index = self.curr_max_index + 1
    self.curr_max_index += 1
    if index % 2 == 0:
      father_index = index / 2
    else:
      father_index = (index-1) / 2
    father_node = self.node_index_dict[father_index]
    node = TreeNode(v)
    if father_node.left:
      father_node.right = node 
    else:
      father_node.left = node
    self.node_index_dict[index] = node
    return father_node.val


  def get_root(self):
    """
    :rtype: TreeNode
    """
    return self.root


  def _build(self, root):
    max_index = 1
    index_dict = {}
    l = []
    l.append((1, root))
    while l:
      index, node = l[0]
      del l[0]
      index_dict[index] = node
      if index > max_index:
        max_index = index
      if node.left:
        l.append((index*2, node.left))
      if node.right:
        l.append((index*2+1, node.right))
    return index_dict, max_index
