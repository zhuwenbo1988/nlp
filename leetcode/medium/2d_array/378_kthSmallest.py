# coding=utf-8

'''

用值进行二分查找

思路非常简单：
1.找出二维矩阵中最小的数left，最大的数right，那么第k小的数必定在left~right之间
2.mid=(left+right) / 2；在二维矩阵中寻找小于等于mid的元素个数count
3.若这个count小于k，表明第k小的数在右半部分且不包含mid，即left=mid+1, right=right，又保证了第k小的数在left~right之间
4.若这个count大于k，表明第k小的数在左半部分且可能包含mid，即left=left, right=mid，又保证了第k小的数在left~right之间
5.因为每次循环中都保证了第k小的数在left~right之间，当left==right时，第k小的数即被找出，等于right

注意：这里的left mid right是数值，不是索引位置。

'''

# https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/

class Solution(object):
  def kthSmallest(self, matrix, k):
    """
    :type matrix: List[List[int]]
    :type k: int
    :rtype: int
    """
    # 最小值
    min_val = matrix[0][0]
    # 最大值
    max_val = matrix[-1][-1]
    while min_val < max_val:
      # 中间值
      mid_val = (min_val+max_val) // 2
      count = self.find(matrix, mid_val, len(matrix), len(matrix))
      if count < k:
        min_val = mid_val + 1
      else:
        max_val = mid_val
    return max_val

  def find(self, matrix, mid_v, row, col):
    count = 0
    # i从最大开始,递减;j从最小开始,递增
    i = row - 1
    j = 0
    while i >= 0 and j < col:
      if matrix[i][j] <= mid_v:
        count += (i+1)
        # 增加j
        j += 1
      else:
        # 减小i
        i += -1
    return count
