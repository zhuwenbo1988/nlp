# coding=utf-8

'''

双指针分别对应奇偶链，最后串起来

'''

# https://leetcode-cn.com/problems/odd-even-linked-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def oddEvenList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
      return
    first_1 = head
    first_2 = head.next
    p1 = first_1
    p2 = first_2
    while p2 and p2.next:
      p1.next = p1.next.next
      p1 = p1.next
      p2.next = p2.next.next
      p2 = p2.next
    p1.next = first_2
    return first_1
