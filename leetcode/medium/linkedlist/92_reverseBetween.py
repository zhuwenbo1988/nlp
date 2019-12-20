# coding=utf-8

'''

一定要用头插法

'''

# https://leetcode-cn.com/problems/reverse-linked-list-ii

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def reverseBetween(self, head, m, n):
    """
    :type head: ListNode
    :type m: int
    :type n: int
    :rtype: ListNode
    """
    if m == n:
      return head
    pre_dump = ListNode(0)
    pre_last = pre_dump
    idx = 1
    # 小于m的部分正常处理
    while idx < m:
      pre_last.next = head
      pre_last = pre_last.next
      head = head.next
      idx += 1
    new_head = ListNode(0)
    # 被反转的链表的最后一个节点，巧妙
    last = head
    # m和n中间的用头插法
    while idx < n+1:
      curr = head
      head = head.next
      # 头插法
      curr.next = new_head.next
      new_head.next = curr
      idx += 1
    # 大于m
    pre_last.next = new_head.next
    last.next = head
    return pre_dump.next
