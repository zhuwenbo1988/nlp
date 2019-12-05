# coding=utf-8

# https://leetcode-cn.com/problems/insertion-sort-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def insertionSortList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head or not head.next:
      return head
    dump = ListNode(0)
    dump.next = head
    while head and head.next:
      if head.val <= head.next.val:
        head = head.next
        continue
      pre = dump
      while pre.next.val < head.next.val:
        pre = pre.next
        continue
      curr = head.next
      head.next = curr.next
      curr.next = pre.next
      pre.next = curr
    return dump.next
