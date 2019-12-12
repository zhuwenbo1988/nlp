# coding=utf-8

'''

快慢指针

'''

# https://leetcode-cn.com/problems/linked-list-cycle/

class Solution(object):
  def hasCycle(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    if not head:
      return False
    slow = head
    fast = head
    while fast.next and fast.next.next:
      slow = slow.next
      fast = fast.next.next
      if slow == fast:
        return True
    return False
