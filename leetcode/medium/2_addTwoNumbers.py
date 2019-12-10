# coding=utf-8

# https://leetcode-cn.com/problems/add-two-numbers/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def addTwoNumbers(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    pre = 0
    head = ListNode(0)
    node = head
    while l1 or l2:
      v1 = 0
      if l1:
        v1 = l1.val
      v2 = 0
      if l2:
        v2 = l2.val
      v = v1 + v2
      v = v + pre
      pre = v / 10
      v = v % 10
      node.next = ListNode(v)
      node = node.next
      if l1:
        l1 = l1.next
      if l2:
        l2 = l2.next
    if pre == 1:
      node.next = ListNode(pre)
      node = node.next
    return head.next
