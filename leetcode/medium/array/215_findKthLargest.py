# coding=utf-8

'''

利用快速排序中的partition思想解决，时间复杂度O(N)

第一部分最难
第二部分还行

'''

# https://leetcode-cn.com/problems/kth-largest-element-in-an-array

class Solution(object):
  def findKthLargest(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    while nums:
      # part 1
      s = 0
      e = len(nums)-1
      # 用v切分整个数组
      v = nums[s]
      while s < e:
        # 从尾部找到第一个小于等于v的数,然后把这个数放到数组的头部
        while s < e and nums[e] > v:
          e += -1
        nums[s] = nums[e]
        # 从头部找到第一个大于v的数,然后填到刚才那个数在尾部的位置
        while s < e and nums[s] <= v:
          s += 1
        nums[e] = nums[s]
      # 把切分值放到切分点
      nums[s] = v
      # part 2
      n = len(nums[s:])
      if n == k:
        return nums[s]
      if n < k:
        nums = nums[:s]
        k = k-n
      else:
        nums = nums[s+1:]
