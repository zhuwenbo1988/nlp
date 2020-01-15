# coding=utf-8

'''

动态规划,需要很烧脑的推导

'''

# https://leetcode-cn.com/problems/unique-binary-search-trees/

# https://leetcode-cn.com/problems/unique-binary-search-trees/solution/bu-tong-de-er-cha-sou-suo-shu-by-leetcode/

class Solution(object):
  def numTrees(self, n):
    """
    :type n: int
    :rtype: int
    """
    dp = [0] * (n+1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n+1):
      for j in range(1, i+1):
        dp[i] += dp[j-1]*dp[i-j]
    return dp[n]
