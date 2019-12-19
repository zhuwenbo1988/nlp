# coding=utf-8

# https://leetcode-cn.com/problems/merge-intervals/

class Solution:
  def merge(self, intervals):
    # 按照区间左端点进行排序
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
      # 不可合并
      if not merged or merged[-1][-1] < interval[0]:
        merged.append(interval)
      else: # 可以合并
        merged[-1][-1] = max(merged[-1][-1], interval[-1])
    return merged 
