# coding=utf-8


'''

https://www.cnblogs.com/yscl/p/10090939.html
https://www.jianshu.com/p/e0a40d6748b8

堆是一颗完全二叉树
堆中的某个结点的值总是大于等于（大顶堆）或小于等于（小顶堆）其孩子结点的值

关键方法：
下沉
如何高效构建堆

次要方法：
获取最小值（或最大值）
删除最小值（或最大值）
替换某个位置的值

堆的应用：
堆排序
合并k个有序数组
无序数据取topk元素


'''

import sys


class MinHeap:
    def __init__(self):
        # 小顶堆的哨兵,堆中的元素索引从1开始
        self._heap = [-sys.maxint]

    def __len__(self):
        return len(self._heap)-1

    def _siftdown(self, i):
        """序号为i的元素下沉"""
        root_i_val = self._heap[i]
        size = len(self)
        while True:
            left_i = i * 2
            if left_i > size:
                break
            if left_i != size and self._heap[left_i] > self._heap[left_i + 1]:
                # left -> right
                left_i += 1
            if root_i_val > self._heap[left_i]:
                self._heap[i] = self._heap[left_i]
            else:
                # 调整完毕
                break
            i = left_i
        self._heap[i] = root_i_val

    def create_heap(self, nums):
        '''
        O(n)复杂度
        当一颗完全二叉树的所有子树都是一个最小堆时，那么这颗树也就是最小堆了。
        因此从最后一个非叶结点开始调整，叶结点本身就是最小堆，不用调整，可以直接跳过。

        :param nums:
        :return:
        '''
        self._heap.extend(nums)
        size = len(self)
        for i in range(size//2, 0, -1):
            self._siftdown(i)

    def pop_min(self):
        """删除堆顶元素"""
        min_val = self._heap[1]
        last = self._heap.pop()
        if len(self) > 0:  # 为空了就不需要向下了
            self._heap[1] = last
            self._siftdown(1)
        return min_val

    def get_min(self):
        if len(self) > 0:
            return self._heap[1]
        return


    def update(self, i, val):
        """更新指定位置的元素, i>=1"""
        if i > len(self) or i < 1:
            return
        self._heap[i] = val
        self._siftdown(i)


class MaxHeap():
    def __init__(self):
        self._heap = [-sys.maxint]

    def __len__(self):
        return len(self._heap)-1

    def _siftdown(self, i):
        root_i_val = self._heap[i]
        size = len(self)
        while True:
            left_i = i * 2
            if left_i > size:
                break
            if left_i != size and self._heap[left_i] < self._heap[left_i + 1]:
                left_i += 1
            if root_i_val < self._heap[left_i]:
                self._heap[i] = self._heap[left_i]
            else:
                break
            i = left_i
        self._heap[i] = root_i_val

    def create_heap(self, nums):
        self._heap.extend(nums)
        n = len(self)
        for i in range(n//2, 0, -1):
            self._siftdown(i)

    def get_max(self):
        return self._heap[1]

    def pop_max(self):
        max_val = self._heap[1]
        last = self._heap.pop()
        if len(self) > 0:
            self._heap[1] = last
            self._siftdown(1)
        return max_val

    def update(self, i, val):
        self._heap[i] = val
        self._siftdown(i)


def heap_sort(nums):
    # heap = MinHeap()
    heap = MaxHeap()
    heap.create_heap(nums)

    result = []
    for i in range(len(heap)):
        # result.append(heap.pop_min())
        result.append(heap.pop_max())
    return result

nums = [1, 4, 56, 2, 5, 9, 1, 0, 0, 4]
print heap_sort(nums)


def top_k(nums, k):
    # heap = MinHeap()
    heap = MaxHeap()
    heap.create_heap(nums[:k])

    for val in nums[k:]:
        # if val > heap.get_min():
        if val < heap.get_max():
            heap.update(1, val)

    result = []
    for i in range(k):
        # result.append(heap.pop_min())
        result.append(heap.pop_max())

    return result

print top_k([1, 2, 5, 2, 4, 10, 7, 1, 3, 5, 9], 3)
print top_k([0.1, 2, -1, 89, 67, 13, 55, 54, 67], 5)