# coding=utf-8

# https://leetcode-cn.com/problems/contains-duplicate/

class Solution(object):
  def containsDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    d = {}
    for n in nums:
      if n in d:
        return True
      d[n] = 1
    return False
