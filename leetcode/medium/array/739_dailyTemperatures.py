# coding=utf-8

'''

维护递减栈，后入栈的元素总比栈顶元素小。

比对当前元素与栈顶元素的大小
若当前元素 小于 栈顶元素：入栈
若当前元素 大于 栈顶元素：弹出栈顶元素，记录两者下标差值即为所求天数，重复此操作直到栈顶元素大于当前元素，然后当前元素入栈
这里用栈记录的是 T 的下标。

很难想到，背吧

'''

# https://leetcode-cn.com/problems/daily-temperatures/

class Solution(object):
  def dailyTemperatures(self, T):
    """
    :type T: List[int]
    :rtype: List[int]
    """
    l = []
    result = [0] * len(T)
    for idx, v in enumerate(T):
      if l:
        while l and T[l[0]] < v:
          i = l[0]
          del l[0]
          result[i] = idx - i
      l.insert(0, idx)
    return result
