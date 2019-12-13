# coding=utf-8

# https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/

class Solution(object):
  def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i = 0
    j = 1
    while j < len(nums):
      if nums[i] == nums[j]:
        j += 1
        continue
      i += 1
      nums[i] = nums[j]
      j += 1
    return i+1 
