# coding=utf-8

'''

从第一个数开始count=1，遇到相同的就加1，遇到不同的就减1，减到0就重新换个数开始计数，总能找到最多的那个

'''

# https://leetcode-cn.com/problems/majority-element/

class Solution(object):
  def majorityElement(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # 取第一个数,计数器设为1
    count = 1
    result = nums[0]
    for i in range(1, len(nums)):
      if result == nums[i]:
        count += 1
      else:
        count += -1
        if count == 0:
          # 取下一个元素,这里要注意,不是当前元素
          result = nums[i+1]
    return result

