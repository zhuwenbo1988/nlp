# coding=utf-8

'''

[]

[1]

[2]
[1,2]

[2] 需要去掉
[1,2] 需要去掉
[2,2]
[1,2,2]

'''

# https://leetcode-cn.com/problems/subsets-ii/

class Solution(object):
  def subsetsWithDup(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    output = [[]]
    for i in range(len(nums)):
        for j in range(len(output)):
            subset = output[j]+[nums[i]]
            subset.sort()
            if subset not in output:
              output.append(subset)
    return output
