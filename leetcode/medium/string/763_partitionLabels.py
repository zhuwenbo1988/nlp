# coding=utf-8

'''

本质上就是合并每个字母的区间,每个字母的区间的结束就是这个字母在字符串中最后出现的位置

'''

# https://leetcode-cn.com/problems/partition-labels/

class Solution(object):
  def partitionLabels(self, S):
    """
    :type S: str
    :rtype: List[int]
    """
    loc_dict = {}
    for i, c in enumerate(S):
      loc_dict[c] = i
    result = []
    begin = 0
    end = 0
    for i, c in enumerate(S):
      end = max(end, loc_dict[c])
      if i == end:
        result.append(end-begin+1)
        begin = i + 1
    return result
