# coding=utf-8

#https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/

class Solution(object):
  def lengthOfLongestSubstring(self, s):
    """
    :type s: str
    :rtype: int
    """
    sub = []
    sub_dict = {}
    max_len = 0
    for c in s:
      if c in sub_dict:
        n = len(sub)
        if n > max_len:
          max_len = n
        i = sub_dict[c]
        sub = sub[i+1:]
        sub_dict = {}
        for i, c1 in enumerate(sub):
          sub_dict[c1] = i
      sub.append(c)
      sub_dict[c] = len(sub) - 1
    n = len(sub)
    if n > max_len:
      max_len = n
    return max_len
