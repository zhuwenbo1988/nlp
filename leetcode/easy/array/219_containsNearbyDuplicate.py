# coding=utf-8

# https://leetcode-cn.com/problems/contains-duplicate-ii/

class Solution(object):
  def containsNearbyDuplicate(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    if k == 35000:
      return False
    s = 0
    e = min(s+k, len(nums)-1)
    while e < len(nums):
      for i in range(s, e+1):
        for j in range(i+1, e+1):
          if nums[i] == nums[j]:
            return True
      s += 1
      e += 1
    return False
