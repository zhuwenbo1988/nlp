# coding=utf-8

# https://leetcode-cn.com/problems/n-repeated-element-in-size-2n-array/

'''

有一个数重复N次，而且所有数的个数为2N，因此，也就是说，要求的这个数占了一半。那么必然可得，一定会出现连续的三个数，其中有两个数相等，也就是所要求的数。举个例子，比如有数组[1, 2, 4, 5, 3, 3, 3, 3]，那么将3打的最散的排列是[1, 3, 2, 3, 4, 3, 5, 3]。可以发现无论怎么排，一定会出现三个连续的数，其中存在两个相同的数。

'''

class Solution(object):
  def repeatedNTimes(self, A):
    """
    :type A: List[int]
    :rtype: int
    """
    for i in range(len(A)-2):
      if A[i] == A[i+1] or A[i] == A[i+2]:
        return A[i]
    return A[len(A)-1]
