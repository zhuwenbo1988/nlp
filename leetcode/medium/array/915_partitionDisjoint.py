# coding=utf-8

'''

完全不会

维护三个变量,一个是当前的切分点,一个是切分点前面数组的最大值,还有一个是实时的最大值,一旦更新切分点,就把左边数组的最大值进行更新

'''

# https://leetcode-cn.com/problems/partition-array-into-disjoint-intervals/

class Solution(object):
  def partitionDisjoint(self, A):
    """
    :type A: List[int]
    :rtype: int
    """
    cut_pos = 0
    left_max = A[0]
    curr_max = A[0]
    for i in range(1, len(A)):
      curr_max = max(curr_max, A[i])
      if A[i] < left_max:
        cut_pos = i
        left_max = curr_max
    return cut_pos+1
