# coding=utf-8

'''

难点:
数组中元素的范围是 [-1000, 1000] ，且整数 k 的范围是 [-1e7, 1e7]。
因为上面这个条件,所以不能用209的方法

如果累计总和，在索引 i 和 j 处相差 k，即 sum[i] - sum[j] = k，则位于索引 i 和 j 之间的元素之和是 k

遍历数组nums，计算从第0个元素到当前元素的和curr_sum，用哈希表保存出现过的curr_sum和它的次数。如果curr_sum - k在哈希表中出现过，则代表从当前下标i往前有连续的子数组的和为k。时间复杂度为O(n)，空间复杂度为O(1)。

'''

# https://leetcode-cn.com/problems/subarray-sum-equals-k

class Solution(object):
  def subarraySum(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    m = {}
    m[0] = 1
    result = 0
    curr_sum = 0
    for n in nums:
      curr_sum += n
      # 如果curr_sum-k有值，则说明从0到某个i（不止一个）的和为curr_sum-k，从这个i到当前下标的和为k
      if curr_sum-k in m:
        result += m[curr_sum-k]
      if curr_sum in m:
        m[curr_sum] += 1
      else:
        m[curr_sum] = 1
    return result
