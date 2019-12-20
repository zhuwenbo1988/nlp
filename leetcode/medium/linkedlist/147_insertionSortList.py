# coding=utf-8

'''

O(n^2)

'''

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
    while head:
      cur = dump
      while cur.next and cur.next.val < head.val:
        cur = cur.next
      node = head
      head = head.next
      node.next = cur.next
      cur.next = node
    return dump.next