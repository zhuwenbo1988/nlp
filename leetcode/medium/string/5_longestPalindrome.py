# coding=utf-8

'''

中心拓展

'''

# https://leetcode-cn.com/problems/longest-palindromic-substring/

class Solution(object):
  def longestPalindrome(self, s):
    """
    :type s: str
    :rtype: str
    """
    final_p = ''
    for i, c in enumerate(s):
      # 以c为中心，重新计算
      curr_p = []
      curr_p.append(s[i])

      # 扩展中心
      for j in range(i+1, len(s)):
        if s[i] != s[j]:
          break
        curr_p.append(s[j])
      b_i = i-1
      e_i = i+len(curr_p)
      # 从中心向两侧扩展
      while b_i > -1 and e_i < len(s):
        if s[b_i] != s[e_i]:
          break
        curr_p.insert(0, s[b_i])
        curr_p.append(s[e_i])
        b_i += -1
        e_i += 1
      # 更新最长回文子串
      if len(curr_p) > len(final_p):
        final_p = ''.join(curr_p)
    return final_p
