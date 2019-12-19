# coding=utf-8

'''

快慢指针找中点，然后递归建树

非递归太复杂

'''

# https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
  def sortedListToBST(self, head):
    """
    :type head: ListNode
    :rtype: TreeNode
    """
    if not head:
      return
    if not head.next:
      return TreeNode(head.val)
    slow = head
    fast = head
    # 不要忘记把慢指针分割出来
    pre = head
    while fast and fast.next:
      pre = slow
      slow = slow.next
      fast = fast.next.next
    pre.next = None
    root = TreeNode(slow.val)
    root.left = self.sortedListToBST(head)
    root.right = self.sortedListToBST(slow.next)
    return root

