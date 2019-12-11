# coding=utf-8

'''

01背包

'''

# https://leetcode-cn.com/problems/word-break/

class Solution(object):
  def wordBreak(self, s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    dp = [False] * (len(s)+1)
    for i in range(1, len(s)+1):
      for word in wordDict:
        if len(word) <= i:
          v1 = s[:i] == word
          if s[:i].endswith(word):
            j = len(s[:i]) - len(word)
          else:
            j = 0
          dp[i] = dp[i] or v1 or dp[j]
    return dp[-1]
