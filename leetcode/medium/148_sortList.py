# coding=utf-8

'''

归并排序
1.快慢指针找切分点(中点)
2.合并时用伪头节点

'''

# https://leetcode-cn.com/problems/sort-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def sortList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
      return head
    if not head.next:
      return head
    mid = self.cut(head)
    left = head
    right = mid.next
    mid.next = None
    return self.merge(self.sortList(left), self.sortList(right))
  
  def cut(self, node):
    if not node:
      return node
    slow = node
    fast = node
    while fast.next and fast.next.next:
      slow = slow.next
      fast = fast.next.next
    return slow

  def merge(self, left, right):
    dump = ListNode(0)
    head = dump
    while left and right:
      if left.val > right.val:
        head.next = right
        right = right.next
      else:
        head.next = left
        left = left.next
      head = head.next
    if left:
      head.next = left
    if right:
      head.next = right
    return dump.next

