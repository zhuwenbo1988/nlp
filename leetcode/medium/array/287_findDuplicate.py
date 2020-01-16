# coding=utf-8

'''

使用环形链表II的方法解题（142.环形链表II）
如果数组中有重复的数，以数组[1,3,4,2,2]为例,我们将数组下标n和数nums[n]建立一个映射关系
0->1
1->3
2->4
3->2
4->2

从下标0出发,以下标0的值为新的下标,可以走出这样一条路径
0 1 3 2 4 2 4 2 4 ...
可以看出,如果把这条路径看做一个链表的话,这就是一个有环的链表,环的入口在下标2,下标2就是重复的元素

所以用题目142的方法
第一步
用快慢指针确定数组是否有重复元素(等同于检测链表是否有环)
slow = nums[slow]
fast = nums[nums[fast]]
第二步
定义两个指针,一个从下标0出发,一个以slow(or fast)为下标出发,直到两个指针相遇,相遇的指针指向的下标就是重复的元素,返回这个指针就行

'''

https://leetcode-cn.com/problems/find-the-duplicate-number/

class Solution(object):
  def findDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    slow = 0
    fast = 0
    slow = nums[slow]
    fast = nums[nums[fast]]
    while slow != fast:
      slow = nums[slow]
      fast = nums[nums[fast]]
    p1 = 0
    # slow=fast
    p2 = slow
    while p1 != p2:
      p1 = nums[p1]
      p2 = nums[p2]
    return p1
