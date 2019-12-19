# coding=utf-8

'''

真牛逼,不会

'''

# https://leetcode-cn.com/problems/subsets/

class Solution(object):
  def subsets(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    output = [[]]
    for i in range(len(nums)):
        for j in range(len(output)):
            output.append(output[j]+[nums[i]])
    return output
