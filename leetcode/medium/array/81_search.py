# coding=utf-8

'''

抄的

'''

# https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/

class Solution(object):
    def search(self, nums, target):
        if not nums:
            return False
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if target == nums[m]:
                return True
            if nums[m] > nums[l]:  # 左边升序
                if target < nums[m] and target >= nums[l]:
                    r = m - 1
                else:
                    l = m + 1
            elif nums[m] < nums[l]:  # 右边升序
                if target > nums[m] and target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
            else:
                l += 1
        return False
