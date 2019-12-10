# coding=utf-8

# https://leetcode-cn.com/problems/partition-equal-subset-sum/

'''
回溯，速度不满足要求

class Solution(object):
  def canPartition(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    result = []
    result.append(0)
    def find(curr, other):
      if result[0] == 1:
        return
      if sum(curr) == sum(other):
        result[0] = 1
        return
      for i in range(len(other)):
        find(curr + [other[i]], other[:i] + other[i+1:])
    find([], nums)
    return result[0] == 1

'''

# https://leetcode-cn.com/problems/partition-equal-subset-sum/solution/0-1-bei-bao-wen-ti-xiang-jie-zhen-dui-ben-ti-de-yo/

class Solution(object):
  def canPartition(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    size = len(nums)

    s = sum(nums)
    # 如果整个数组的和都不是偶数，就无法平分
    if s % 2 == 1:
      return False
    
    # 背包的容量
    target = s // 2
    # 动态规划的结果是判断 
    dp = [[False] * (target+1) for _ in range(size)]
 
    # 初始化
    for i in range(target+1):
      if nums[0] != i:
        dp[0][i] = False
      else:
        dp[0][i] = True
    
    for i in range(1, size):
      for j in range(target+1):
        # 超过当前的容量了
        if nums[i] > j:
          dp[i][j] = dp[i-1][j]
        else: # 没超过
          # 上一个i已经满足当前的j，或者
          if dp[i - 1][j] or dp[i - 1][j - nums[i]]:
            dp[i][j] = True
          else:
            dp[i][j] = False
    return dp[-1][-1]
