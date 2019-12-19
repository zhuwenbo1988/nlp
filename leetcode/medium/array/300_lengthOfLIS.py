# coding=utf-8

'''

    # 动态规划的思路：将 dp 数组定义为：以 nums[i] 结尾的最长上升子序列的长度
    # 那么题目要求的，就是这个 dp 数组中的最大者
    # 以数组  [10, 9, 2, 5, 3, 7, 101, 18] 为例：
    # dp 的值： 1  1  1  2  2  3  4    4

'''

# https://leetcode-cn.com/problems/longest-increasing-subsequence

class Solution(object):
  def lengthOfLIS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
      return 0
    dp = [1] * (len(nums))
    for i in range(1, len(nums)):
      for j in range(i):
        if nums[i] > nums[j]:
          dp[i] = max(dp[i], dp[j]+1)
    return max(dp)
