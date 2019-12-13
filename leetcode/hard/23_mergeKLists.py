# coding=utf-8

# https://leetcode-cn.com/problems/merge-k-sorted-lists

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
  def mergeKLists(self, lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    if not lists:
      return
    n = len(lists)
    return self.merge(lists, 0, n-1)
  def merge(self, lists, left, right):
    if left == right:
      return lists[left]
    mid = (left + right) // 2
    list_1 = self.merge(lists, left, mid)
    list_2 = self.merge(lists, mid+1, right)
    return self.merge_two_lists(list_1, list_2)
  def merge_two_lists(self, list_1, list_2):
    if not list_1:
      return list_2
    if not list_2:
      return list_1
    if list_1.val < list_2.val:
      list_1.next = self.merge_two_lists(list_1.next, list_2)
      return list_1
    else:
      list_2.next = self.merge_two_lists(list_1, list_2.next)
      return list_2
