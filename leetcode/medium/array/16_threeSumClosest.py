# coding=utf-8

'''

先排序, 然后遍历, 然后内部使用双指针, 时间复杂度应该是O(n²)

'''

 # https://leetcode-cn.com/problems/3sum-closest

import sys

class Solution(object):
  def threeSumClosest(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    # 必须要排序
    nums.sort()

    min_gap = sys.maxint
    min_sum = 0
    for i in range(len(nums)):
      s = i+1
      e = len(nums)-1
      # 双指针
      while s < e:
        three_sum = nums[i] + nums[s] + nums[e]
        gap = abs(target - three_sum)
        if gap < min_gap:
          min_gap = gap
          min_sum = three_sum
        if three_sum == target:
          return three_sum
        if three_sum < target:
          s += 1
        else:
          e += -1
    return min_sum