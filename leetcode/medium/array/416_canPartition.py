# coding=utf-8

# https://leetcode-cn.com/problems/partition-equal-subset-sum/

# https://leetcode-cn.com/problems/partition-equal-subset-sum/solution/0-1-bei-bao-wen-ti-xiang-jie-zhen-dui-ben-ti-de-yo/

'''

0-1背包
背吧

'''

class Solution(object):
  def canPartition(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    s = sum(nums)
    # 如果整个数组的和都不是偶数，就无法平分
    if s % 2 == 1:
      return False
    # 背包的容量
    target = s // 2
    dp = [False] * (target+1)
    dp[0] = True
    for v in nums:
      for i in range(target, v-1, -1):
        if dp[i-v] == True:
          dp[i] = True
    return dp[target]

'''

会超时，但是容易理解

'''

class Solution(object):
  def canPartition(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    s = sum(nums)
    # 如果整个数组的和都不是偶数，就无法平分
    if s % 2 == 1:
      return False
    # 背包的容量
    target = s // 2
    result = []
    def find(tmp, other):
      if sum(tmp) == target:
        result.append(tmp)
      if sum(tmp) > target:
        return
      for i in range(len(other)):
        find(tmp+[other[i]], other[i+1:])
    find([], nums)
    return len(result) != 0