# coding=utf-8

'''

原地删除肯定是双指针，一个指向遍历的元素，一个指向可以写入的位置

'''

# https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/

class Solution(object):
  def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    k = 2
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
