# coding=utf-8

'''

不重复元素的左边有偶数个元素,设为A,右边也有偶数个元素,设为B
A中元素,偶数索引的元素与偶数索引+1的元素相等
B中元素,偶数索引的元素与偶数索引+1的元素不等
所以使用二分搜索检验偶数索引,如果中点为奇数索引,则减一



'''

# https://leetcode-cn.com/problems/single-element-in-a-sorted-array/

class Solution(object):
  def singleNonDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    s = 0
    e = len(nums)-1
    while s < e:
      mid = (s+e) // 2
      if mid % 2 == 1:
        mid += -1
      if nums[mid] == nums[mid+1]:
        s = mid + 2
      else:
        e = mid
    # s==e
    return nums[s]

'''

易理解但是啰嗦的版本

'''

class Solution(object):
  def singleNonDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    s = 0
    e = len(nums)-1
    while s < e:
      mid = (s+e) // 2
      if mid % 2 == 1:
        if nums[mid] == nums[mid+1]:
          e = mid
        else:
          s = mid+1
      else:
        if nums[mid] == nums[mid+1]:
          s = mid+1
        else:
          e = mid
    # s==e
    return nums[s]