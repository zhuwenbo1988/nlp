# coding=utf-8

# https://leetcode-cn.com/problems/linked-list-cycle-ii/

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def detectCycle(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    # 先判断是否有环
    has_cycle = False
    slow = head
    fast = head
    # 快慢指针判断是否有环
    while fast and fast.next:
      slow = slow.next
      fast = fast.next.next
      if slow == fast:
        has_cycle = True
        break
    if has_cycle:
      '''
      首先假定链表起点到入环的第一个节点A的长度为a【未知】，到快慢指针相遇的节点B的长度为（a + b）【这个长度是已知的】。现在我们想知道a的值，注意到快指针p2始终是慢指针p走过长度的2倍，所以慢指针p从B继续走（a + b）又能回到B点，如果只走a个长度就能回到节点A。但是a的值是不知道的，解决思路是曲线救国，注意到起点到A的长度是a，那么可以用一个从起点开始的新指针q和从节点B开始的慢指针p同步走，相遇的地方必然是入环的第一个节点A。 
      如何理解慢指针只走a个长度就能回到节点A：假设环内剩余长度为c,则快指针走过的长度为: 2(a+b) = a + b + c + b 所以c=a
      '''
      while slow != head:
        slow = slow.next
        head = head.next
      return slow
    else: # 没有环
      return
