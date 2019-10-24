# coding=utf-8

# https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def removeNthFromEnd(self, head, n):
    """
    :type head: ListNode
    :type n: int
    :rtype: ListNode
    """
    nodes = []
    while head:
      tmp_node = head
      i = 1
      while tmp_node.next:
        tmp_node = tmp_node.next
        i += 1
      if i == n:
        if len(nodes) == 0:
          nodes.append(head.next)
        else:
          pre_node = nodes[-1]
          pre_node.next = head.next
        break
      nodes.append(head)
      head = head.next
    return nodes[0]
