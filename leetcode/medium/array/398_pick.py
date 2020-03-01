# coding=utf-8

'''

蓄水池采样算法

'''

from random import randint

class Solution(object):

  def __init__(self, nums):
    """
    :type nums: List[int]
    """
    self.nums = nums

  def pick(self, target):
    """
    :type target: int
    :rtype: int
    """
    n = 0
    final_index = 0
    for i in range(len(self.nums)):
      if self.nums[i] != target:
        continue
      n += 1
      if randint(1, n) % n + 1 == 1:
        final_index = i
    return final_index
