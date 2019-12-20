# coding=utf-8

# https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/

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
      # 遇到重复字符了
      if c in sub_dict:
        if len(sub) > max_len:
          max_len = len(sub)
        # i是c在sub中的位置
        i = sub_dict[c]
        # 从sub，重新在sub_dict中设置每个字符的位置
        sub = sub[i+1:]
        sub_dict = {}
        for i, c1 in enumerate(sub):
          sub_dict[c1] = i
      sub.append(c)
      # 存储c在sub中的位置
      sub_dict[c] = len(sub) - 1
    # 别忘了比较最后一个子串
    if len(sub) > max_len:
      max_len = len(sub)
    return max_len