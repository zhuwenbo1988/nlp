# coding=utf-8

'''

说是循环旋转，但其实本质上是将尾部向前数第K个元素作为头，原来的头接到原来的尾上

'''

# https://leetcode-cn.com/problems/rotate-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def rotateRight(self, head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    """
    if not head:
      return head
    if not head.next:
      return head
    length = 1
    end = head
    # 先计算链表的长度
    while end.next:
      length += 1
      end = end.next
    # 计算切分的位置
    n = k % length
    if n == 0:
      return head
    n = length - n
    m = 1
    cut = head
    while cut:
      if m == n:
        break
      m += 1
      cut = cut.next
    # 新的表头
    new_head = cut.next
    # 新的表尾
    cut.next = None
    # 两段连接起来
    end.next = head
    return new_head
