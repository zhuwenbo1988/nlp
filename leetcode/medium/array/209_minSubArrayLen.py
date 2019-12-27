# coding=utf-8

'''

因为是连续子序列,所以用双指针来滑动窗口

不要用数组来装中间结果,会超时

数组和给出的数都是正整数,使得这题能用这种方法

'''

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
      while curr_sum < s and r < len(nums):
        curr_sum += nums[r]
        r += 1
      while curr_sum >= s and l >= 0:
        result = min(result, r-l)
        curr_sum -= nums[l]
        l += 1
    return result
