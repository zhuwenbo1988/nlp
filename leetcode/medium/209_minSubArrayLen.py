# coding=utf-8

# https://leetcode-cn.com/problems/minimum-size-subarray-sum/

class Solution(object):
  def minSubArrayLen(self, s, nums):
    """
    :type s: int
    :type nums: List[int]
    :rtype: int
    """
    if s > sum(nums):
      return 0
    l = 0
    r = 0
    result = len(nums)
    curr_sum = 0
    while r < len(nums):
      print curr_sum
      while curr_sum < s and r < len(nums):
        curr_sum += nums[r]
        r += 1
      while curr_sum >= s and l >= 0:
        result = min(result, r-l)
        curr_sum -= nums[l]
        l += 1
    return result
