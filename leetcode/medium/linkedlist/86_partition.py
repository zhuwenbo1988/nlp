# coding=utf-8

# https://leetcode-cn.com/problems/partition-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def partition(self, head, x):
    """
    :type head: ListNode
    :type x: int
    :rtype: ListNode
    """
    dump_1 = ListNode(0)
    first_1 = dump_1
    dump_2 = ListNode(0)
    first_2 = dump_2
    while head:
      node = head
      head = head.next
      node.next = None
      if node.val < x:
        dump_1.next = node
        dump_1 = dump_1.next
      else:
        dump_2.next = node
        dump_2 = dump_2.next
    dump_1.next = first_2.next
    return first_1.next
