# coding=utf-8

'''

完全不会

'''

# https://leetcode-cn.com/problems/palindrome-partitioning/

class Solution(object):
  def partition(self, s):
    """
    :type s: str
    :rtype: List[List[str]]
    """
    result = []
    def find(sub_s, tmp):
      if sub_s == sub_s[::-1]:
        # 不return
        result.append(tmp + [sub_s])
      for i in range(1, len(sub_s)):
        if sub_s[:i] == sub_s[:i][::-1]:
          find(sub_s[i:], tmp + [sub_s[:i]])
    find(s, [])
    return result
