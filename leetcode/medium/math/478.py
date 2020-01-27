# coding=utf-8

'''

方法一：拒绝采样

我们使用一个边长为 2R 的正方形覆盖住圆 C，并在正方形内随机生成点，若该点落在圆内，我们就返回这个点，否则我们拒绝这个点，重新生成知道新的机点落在圆内。

'''

# https://leetcode-cn.com/problems/generate-random-point-in-a-circle/

from random import uniform
import math

class Solution(object):

  def __init__(self, radius, x_center, y_center):
    """
    :type radius: float
    :type x_center: float
    :type y_center: float
    """
    self.r = radius
    self.x = x_center
    self.y = y_center      


  def randPoint(self):
    """
    :rtype: List[float]
    """
    x1 = self.x - self.r
    x2 = self.x + self.r
    y1 = self.y - self.r
    y2 = self.y + self.r
    
    x = 0
    y = 0
    while True:
      x = uniform(x1, x2)
      y = uniform(y1, y2)
      dist = math.sqrt((self.x - x)**2 + (self.y - y)**2)
      if dist <= self.r:
        break

    return [x, y]
