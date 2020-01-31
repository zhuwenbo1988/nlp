# coding=utf-8

'''

这个题目开头暗示了n的范围，所以可以加以利用，将元素转换成数组的索引并对应的将该处的元素乘以-1；

若数组索引对应元素的位置本身就是负数，则表示已经出现过一次；在结果列表里增加该索引的正数就行；

'''

# https://leetcode-cn.com/problems/find-all-duplicates-in-an-array/

class Solution(object):
  def findDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    n = len(nums)
    result = []
    for i in range(n):
      j = abs(nums[i]) - 1
      if nums[j] < 0:
        result.append(abs(nums[i]))
      nums[j] *= -1
    return result