# coding=utf-8

# https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/

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
    first = head
    while head:
      count = 1
      node = head
      while node.next:
        if node.val != node.next.val:
          break
        count += 1
        node = node.next
      if count > 1:
        head.next = node.next
      head = head.next
    return first
