# coding=utf-8

'''

真牛逼,不会

这样理解:
初始化:
[]
外层循环第一次:
[1]
外层循环第二次:
[2]
[1,2]
外层循环第三次:
[3]
[1,3]
[2,3]
[1,2,3]

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
