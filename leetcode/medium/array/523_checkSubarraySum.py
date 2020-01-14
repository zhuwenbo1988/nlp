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
      if k != 0:
        curr_sum = curr_sum % k
      if curr_sum in m:
        if i - m[curr_sum] > 1:
          return True
      else:
        m[curr_sum] = i
    return False
