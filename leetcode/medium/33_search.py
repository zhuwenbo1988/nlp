# coding=utf-8

'''

每次找到有序的区间查看 target 是否在里面,如果在里面直接在这区间寻找
如果不在则继续从另一边不是有序的区间继续二分
时间复杂度 O(log(n)) 

'''


# https://leetcode-cn.com/problems/search-in-rotated-sorted-array/

class Solution(object):
  def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    i = 0
    j = len(nums) - 1
    while i <= j:
      idx = (i + j) // 2
      if nums[idx] == target:
        return idx
      a = nums[i]
      b = nums[j]
      c = nums[idx]
      if a > c:
        if target > c and target <= b:
          i = idx + 1
        else:
          j = idx - 1
      else:
        if target >= a and target < c:
          j = idx - 1
        else:
          i = idx + 1
    return -1
