# coding=utf-8

'''

切分出前半段和后半段,再把后半段插进前半段

'''

# https://leetcode-cn.com/problems/reorder-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def reorderList(self, head):
    """
    :type head: ListNode
    :rtype: None Do not return anything, modify head in-place instead.
    """
    if not head:
      return head
      
    # 快慢指针找中点
    slow = head
    fast = head
    while fast and fast.next:
      slow = slow.next
      fast = fast.next.next
    
    # 翻转链表
    node = slow.next
    dump = ListNode(0)
    while node:
      tmp = node
      node = node.next
      tmp.next = dump.next
      dump.next = tmp
    # 截取前半段和后半段
    right = dump.next
    left = head
    slow.next = None

    # 将后半段插入前半段
    while right:
      tmp = right
      right = right.next
      tmp.next = left.next
      left.next = tmp
      left = left.next.next
      
    return head
