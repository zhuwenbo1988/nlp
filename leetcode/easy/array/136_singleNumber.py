# coding=utf-8

'''

位运算 - 异或

当然，异或运算会有溢出的风险

'''

# https://leetcode-cn.com/problems/single-number/

# https://leetcode-cn.com/problems/single-number/solution/zhi-chu-xian-yi-ci-de-shu-zi-by-leetcode/

class Solution(object):
  def singleNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    a = 0
    for val in nums:
      a = a ^ val
    return a
