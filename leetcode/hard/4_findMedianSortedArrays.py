# coding=utf-8

'''

太复杂了

'''

# https://leetcode-cn.com/problems/median-of-two-sorted-arrays

class Solution(object):
  def findMedianSortedArrays(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    def find(_nums1, _nums2, k):
      m = len(_nums1)
      n = len(_nums2)
      if m > n:
        # 保证左边数量少，右边数量多
        return find(_nums2, _nums1, k)
      # 
      if not _nums1:
        return _nums2[k-1]
      #
      if k == 1:
        return min(_nums1[0], _nums2[0])
      i = min(k // 2, m) - 1
      j = min(k // 2, n) - 1
      if _nums1[i] > _nums2[j]:
        return find(_nums1, _nums2[j+1:], k-j-1)
      else:
        return find(_nums1[i+1:], _nums2, k-i-1)
    m = len(nums1)
    n = len(nums2)
    left = (m+n+1) // 2
    right = (m+n+2) // 2
    left_max = find(nums1, nums2, left)
    right_min = find(nums1, nums2, right)
    medium = (left_max + right_min) / 2.0
    return medium
