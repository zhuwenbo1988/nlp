# coding=utf-8

# https://leetcode-cn.com/problems/swap-nodes-in-pairs/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def swapPairs(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    tmp_node = ListNode(-1)
    first = tmp_node
    while True:
      pre = head
      if not pre:
        break
      curr = head.next
      if not curr:
        tmp_node.next = pre
        break
      head = curr.next
      pre.next = curr.next
      curr.next = pre
      tmp_node.next = curr
      tmp_node = pre
    return first.next
