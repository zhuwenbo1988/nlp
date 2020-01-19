# coding=utf-8

'''

水平扫描

'''

# https://leetcode-cn.com/problems/longest-common-prefix/

class Solution(object):
  def longestCommonPrefix(self, strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if not strs:
      return ''
    prefix = strs[0]
    for s in strs:
      while not s.startswith(prefix):
        prefix = prefix[:len(prefix)-1]
    return prefix
