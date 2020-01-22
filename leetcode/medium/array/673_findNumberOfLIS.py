# coding=utf-8

# https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/

class Solution(object):
  def findNumberOfLIS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
      return 0
    m = len(nums)
    dp_len = [1] * m
    dp_n = [1] * m
    for i in range(1, m):
      for j in range(i):
        if nums[i] > nums[j]:
          if dp_len[j]+1 == dp_len[i]:
            dp_n[i] += dp_n[j]
          if dp_len[j]+1 > dp_len[i]:
            dp_len[i] = dp_len[j]+1
            dp_n[i] = dp_n[j]
    max_len = max(dp_len)
    result = 0
    for length, count in zip(dp_len, dp_n):
      if length == max_len:
        result += count
    return result
