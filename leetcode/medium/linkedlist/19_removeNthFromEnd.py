# coding=utf-8

# https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def removeNthFromEnd(self, head, n):
    """
    :type head: ListNode
    :type n: int
    :rtype: ListNode
    """
    dump = ListNode(0)
    first = dump
    while head:
      i = 1
      tmp = head
      while tmp.next:
        i += 1
        tmp = tmp.next
      if i == n:
        dump.next = head.next
        break
      dump.next = head
      dump = dump.next
      head = head.next
    return first.next