# coding=utf-8

'''

哈希表

'''

# https://leetcode-cn.com/problems/continuous-subarray-sum/

class Solution(object):
  def checkSubarraySum(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    curr_sum = 0
    m = {}
    m[0] = -1
    for i in range(len(nums)):
      curr_sum += nums[i]
      # 保险的判断
      if k != 0:
        curr_sum = curr_sum % k
      if curr_sum in m:
        # 连续子序列长度大于等于2
        if i - m[curr_sum] > 1:
          return True
      else:
        m[curr_sum] = i
    return False
