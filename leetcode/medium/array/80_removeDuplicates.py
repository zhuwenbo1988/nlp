# coding=utf-8

# https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/

class Solution(object):
  def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    k = 2
    if len(nums) < k:
      return len(nums)
    cnt = 1
    length_idx = 1
    for i in range(1, len(nums)):
      if nums[i] == nums[i-1]:
        if cnt < k:
          nums[length_idx] = nums[i]
          cnt += 1
          length_idx += 1
      else:
        nums[length_idx] = nums[i]
        cnt = 1
        length_idx += 1
    return length_idx
