# coding=utf-8

# https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def deleteDuplicates(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    pre_dump = ListNode(0)
    pre_dump.next = head
    first = pre_dump
    while head:
      i = 0
      while head.next:
        if head.val != head.next.val:
          break
        head = head.next
        i += 1
      # head的值是存在重复的，全部删去
      if i > 0:
        pre_dump.next = head.next
      else:
        pre_dump = head
      head = head.next
    return first.next
