# coding=utf-8

'''

利用快速排序中的partition思想解决，时间复杂度O(N)

我喜欢
'''

# https://leetcode-cn.com/problems/kth-largest-element-in-an-array

class Solution(object):
  def findKthLargest(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    while nums:
      s = 0
      e = len(nums)-1
      v = nums[s]
      while s < e:
        while s < e and nums[e] > v:
          e += -1
        nums[s] = nums[e]
        while s < e and nums[s] <= v:
          s += 1
        nums[e] = nums[s]
      nums[s] = v
      i = s
      n = len(nums[i:])
      if n == k:
        return nums[i]
      if n < k:
        nums = nums[:i]
        k = k-n
      else:
        nums = nums[i+1:]
