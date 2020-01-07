# coding=utf-8

# https://leetcode-cn.com/problems/3sum/

'''

三指针

https://leetcode-cn.com/problems/3sum/solution/pai-xu-shuang-zhi-zhen-zhu-xing-jie-shi-python3-by/

'''

class Solution(object):
  def threeSum(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    result = []
    nums.sort()
    n = len(nums)
    for i in range(n):
      if i > 0 and nums[i] == nums[i-1]:
        continue
      left = i + 1
      right = n - 1
      while left < right:
        s = nums[i] + nums[left] + nums[right]
        if s == 0:
          result.append((nums[i], nums[left], nums[right]))
          while left < right and nums[left] == nums[left+1]:
            left += 1
          while left < right and nums[right] == nums[right-1]:
            right += -1
          left += 1
          right += -1
        elif s > 0:
          right += -1
        else:
          left += 1
    return result