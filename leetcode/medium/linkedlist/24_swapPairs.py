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
      # 没有第一个节点
      if not pre:
        break
      curr = head.next
      # 没有第二个节点
      if not curr:
        tmp_node.next = pre
        break
      # head置于下一个要处理的节点
      head = curr.next
      # 两两交换
      pre.next = curr.next
      curr.next = pre
      tmp_node.next = curr
      # tmp_node置于下一个要处理的节点的前一个节点
      tmp_node = pre
    return first.next
