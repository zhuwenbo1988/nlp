# coding=utf-8

# https://leetcode-cn.com/problems/add-two-numbers-ii/

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
    s1 = []
    s2 = []
    while l1:
      s1.insert(0, l1.val)
      l1 = l1.next
    while l2:
      s2.insert(0, l2.val)
      l2 = l2.next
    pre = 0
    result = []
    while s1 or s2:
      v1 = 0
      if s1:
        v1 = s1[0]
        del s1[0]
      v2 = 0
      if s2:
        v2 = s2[0]
        del s2[0]
      v = v1 + v2
      v = v + pre
      pre = v / 10
      v = v % 10
      result.append(v)
    if pre == 1:
      result.append(pre)
    dump = ListNode(0)
    first = dump
    for v in result[::-1]:
      node = ListNode(v)
      dump.next = node
      dump = dump.next
    return first.next
