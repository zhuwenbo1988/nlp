# coding=utf-8

'''

贪心
该算法通用且简单：遍历数组并在每个步骤中更新：
当前元素
当前元素位置的最大和
迄今为止的最大和

'''

# https://leetcode-cn.com/problems/maximum-subarray/

class Solution(object):
  def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    dp = [0] * len(nums)
    dp[0] = nums[0]
    for i in range(1, len(nums)):
      dp[i] = max(dp[i-1]+nums[i], nums[i])
    return max(dp)

'''

在本题的基础上,找到最大和连续子数组的起始位置和终止位置

'''

class Solution(object):
  def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    dp = [0] * len(nums)
    dp[0] = nums[0]
    begin = 0
    end = 0
    max_b = 0
    max_e = 0
    max_dp = dp[0]
    for i in range(1, len(nums)):
      if dp[i-1]+nums[i] > nums[i]:
        dp[i] = dp[i-1]+nums[i]
        end = i
      else:
        dp[i] = nums[i]
        begin = i
        end = i
      if max_dp < dp[i]:
        max_dp = dp[i]
        max_b = begin
        max_e = end
    # return max(dp)
    # return max_dp
    return sum(nums[max_b:max_e+1])