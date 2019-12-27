# coding=utf-8

# https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/

class Solution(object):
  def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    k = 1
    if len(nums) < k:
      return len(nums)
    cnt = 1
    new_len = 1
    # 从第一个元素开始,当前元素与上一个元素进行比较
    for i in range(1, len(nums)):
      if nums[i] == nums[i-1]:
        if cnt < k:
          nums[new_len] = nums[i]
          cnt += 1
          new_len += 1
      else:
        nums[new_len] = nums[i]
        cnt = 1
        new_len += 1
    return new_len