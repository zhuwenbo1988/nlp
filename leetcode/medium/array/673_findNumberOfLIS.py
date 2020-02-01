# coding=utf-8

'''

用动态规划计算最长递增子序列的长度时要同时计算当前这个最长长度的个数
计算个数时要注意，一个最长长度是有多种情况的，并且是前面个数的累加

'''

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
          # 核心
          # 更新个数
          if dp_len[j]+1 == dp_len[i]:
            dp_n[i] += dp_n[j]
          if dp_len[j]+1 > dp_len[i]:
            dp_n[i] = dp_n[j]
            # 更新长度
            dp_len[i] = dp_len[j] + 1
    max_len = max(dp_len)
    result = 0
    for length, count in zip(dp_len, dp_n):
      if length == max_len:
        result += count
    return result
