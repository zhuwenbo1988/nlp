# coding=utf-8

'''

我们首先把两棵树的根节点入栈，栈中的每个元素都会存放两个根节点，并且栈顶的元素表示当前需要处理的节点。在迭代的每一步中，我们取出栈顶的元素并把它移出栈，并将它们的值相加。随后我们分别考虑这两个节点的左孩子和右孩子，如果两个节点都有左孩子，那么就将左孩子入栈；如果只有一个节点有左孩子，那么将其作为第一个节点的左孩子；如果都没有左孩子，那么不用做任何事情。对于右孩子同理。

'''

# https://leetcode-cn.com/problems/merge-two-binary-trees/

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def mergeTrees(self, t1, t2):
    """
    :type t1: TreeNode
    :type t2: TreeNode
    :rtype: TreeNode
    """
    if not t1:
      return t2
    l = []
    l.append((t1, t2))
    while l:
      n1, n2 = l[0]
      del l[0]
      if not n1 or not n2:
        continue
      n1.val += n2.val
      if n1.left:
        l.append((n1.left, n2.left))
      else:
        # 巧妙
        n1.left = n2.left
      if n1.right:
        l.append((n1.right, n2.right))
      else:
        # 巧妙
        n1.right = n2.right
    return t1

