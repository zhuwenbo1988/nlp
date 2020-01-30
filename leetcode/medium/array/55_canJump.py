# coding=utf-8

'''

利用贪心算法求求解。因为只要前面的数字所能跳跃的最大步数大于等于数组长度那么就一定可以跳到最后一个位置

'''

# https://leetcode-cn.com/problems/jump-game

class Solution:
    def canJump(self, nums):
        end = 0
        n = len(nums)
        for i in range(n-1):
          # 非常重要的判断条件
          if end >= i:
            end = max(end, i + nums[i])
        return end >= n-1
