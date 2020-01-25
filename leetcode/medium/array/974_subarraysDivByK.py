# coding=utf-8

'''

连续子数组问题,使用前缀和
和为K的连续子数组,使用前缀和的差来解决
本题
若i到j是能被K整除的连续子数组,即(sum[j]-sum[i])%K=0,则sum[i]和sum[j]是同余的
换句话说只要sum[j]有同余的sum[i],则从i到j的连续子数组是可以被K整除的

'''

# https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/

class Solution(object):
  def subarraysDivByK(self, A, K):
    """
    :type A: List[int]
    :type K: int
    :rtype: int
    """
    d = {}
    d[0] = 1
    curr_sum = 0
    result = 0
    for i in range(len(A)):
      curr_sum += A[i]
      key = curr_sum % K
      if key in d:
        result += d[key]
        d[key] += 1
      else:
        d[key] = 1
    return result
