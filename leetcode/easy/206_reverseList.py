# coding=utf-8

# https://leetcode-cn.com/problems/reverse-linked-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def reverseList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    node = ListNode(0)
    while head:
      # 记住这个顺序
      tmp = head
      head = head.next
      tmp.next = node.next
      node.next = tmp
    return node.next
