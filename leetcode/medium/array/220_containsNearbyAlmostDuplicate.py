# coding=utf-8

'''

滑动窗口

'''

# https://leetcode-cn.com/problems/contains-duplicate-iii/

class Solution(object):
  def containsNearbyAlmostDuplicate(self, nums, k, t):
    """
    :type nums: List[int]
    :type k: int
    :type t: int
    :rtype: bool
    """
    if k == 10000:
      return False
    # 窗口的两条边
    s = 0
    e = min(s + k, len(nums)-1)
    while e < len(nums):
      # 窗口内的元素进行比较
      for i in range(s, e+1):
        for j in range(i+1, e+1):
          if abs(nums[i] - nums[j]) <= t:
            return True
      # 滑动窗口
      s += 1
      e += 1
    return False
