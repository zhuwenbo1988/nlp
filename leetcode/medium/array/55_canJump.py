# coding=utf-8

'''

利用贪心算法求求解。因为只要前面的数字所能跳跃的最大步数大于等于数组长度那么就一定可以跳到最后一个位置

'''

# https://leetcode-cn.com/problems/jump-game

class Solution:
    def canJump(self, nums):
        start = 0
        end = 0
        n = len(nums)
        while start <= end and end < len(nums) - 1:
            end = max(end, nums[start] + start)
            start += 1
        return end >= n - 1
