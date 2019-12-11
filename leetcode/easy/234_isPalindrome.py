# coding=utf-8

# https://leetcode-cn.com/problems/palindrome-linked-list/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def isPalindrome(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    if not head:
      return True
    slow = head
    fast = head
    pre = None
    # 快慢指针找中点,奇数个节点找的是中点,偶数个节点找的是中间两个点左边的
    while fast.next and fast.next.next:
      slow = slow.next
      fast = fast.next.next
    # 反转链表
    dump = ListNode(0)
    while slow:
      tmp = slow
      slow = slow.next
      tmp.next = dump.next
      dump.next = tmp
    pre = dump.next
    # 比较
    # 不用担心后半段链表比前面长,因为head的结尾是slow,前半段和后半段的长度是一样的
    while head and pre:
      if head.val != pre.val:
        return False
      head = head.next
      pre = pre.next
    return True
