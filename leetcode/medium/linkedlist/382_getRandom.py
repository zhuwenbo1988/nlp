# coding=utf-8

'''

思路来自k=1的蓄水池采样算法

蓄水池算法适用于对一个不清楚规模的数据集进行采样

'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

from random import randint

class Solution(object):

  def __init__(self, head):
    """
    @param head The linked list's head.
    Note that the head is guaranteed to be not null, so it contains at least one node.
    :type head: ListNode
    """
    self.head = head

  def getRandom(self):
    """
    Returns a random node's value.
    :rtype: int
    """
    # 蓄水池算法，k=1
    i = 2
    final_val = self.head.val
    curr = self.head.next
    while curr:
      # 第 i 节点以 1/i 概率被选中并替换当前值
      if randint(1, i) % i + 1 == 1:
        final_val = curr.val
      curr = curr.next
      i += 1
    return final_val
