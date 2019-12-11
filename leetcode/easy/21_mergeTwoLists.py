# coding=utf-8

# https://leetcode-cn.com/problems/merge-two-sorted-lists/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def mergeTwoLists(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    dump = ListNode(0)
    head = dump
    while l1 and l2:
      v1 = l1.val
      v2 = l2.val
      if v1 < v2:
        dump.next = ListNode(v1)
        l1 = l1.next
      else:
        dump.next = ListNode(v2)
        l2 = l2.next
      dump = dump.next
    if l1:
      dump.next = l1
    if l2:
      dump.next = l2
    return head.next
