# coding=utf-8

# https://leetcode-cn.com/problems/3sum/

'''
超时
'''
class Solution(object):
  def threeSum(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    result = []
    unique_map = {}
    for i in range(len(nums)):
      for j in range(i+1, len(nums)):
        a = nums[i]
        b = nums[j]
        other = []
        other.extend(nums[i+1:j])
        other.extend(nums[j+1:])
        l = []
        for c in other:
          if a+b+c == 0:
            pair = [a, b, c]
            pair.sort()
            key = '-'.join([str(n) for n in pair])
            if key in unique_map:
              continue
            l.append([a, b, c])
            unique_map[key] = 1
        result.extend(l)
    return result
