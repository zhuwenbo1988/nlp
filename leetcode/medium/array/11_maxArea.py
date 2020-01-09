# coding=utf-8

'''

双指针

'''
# https://leetcode-cn.com/problems/container-with-most-water/
# https://cloud.tencent.com/developer/news/440678

class Solution(object):
  def maxArea(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    # 一头一尾
    s = 0
    e = len(height) - 1
    max_area = 0
    while s < e:
      b = e - s
      # 1.计算h
      # 2.谁小谁就更新一位
      if height[s] < height[e]:
        h = height[s]
        s += 1
      else:
        h = height[e]
        e += -1
      area = b * h
      if area > max_area:
        max_area = area
    return max_area